from numpywren import frontend
from numpywren.frontend import *
import ast
import astor
import dill
import logging
import time
from numpywren.matrix import BigMatrix
from numpywren.matrix_init import shard_matrix
from numpywren import exceptions, compiler, utils
import asyncio
import numpy as np
from numpywren.kernels import *
from collections import namedtuple
import inspect
from operator import *
from numpywren import lambdapack as lp
from scipy.linalg import qr
import sympy
from numbers import Number
import copy

def lpcompile(function):
    function_ast = ast.parse(inspect.getsource(function)).body[0]
    logging.debug("Python AST:\n{}\n".format(astor.dump(function_ast)))
    parser = frontend.LambdaPackParse()
    type_checker = frontend.LambdaPackTypeCheck()
    lp_ast = parser.visit(function_ast)
    logging.debug("IR AST:\n{}\n".format(astor.dump_tree(lp_ast)))
    lp_ast_type_checked = type_checker.visit(lp_ast)
    logging.debug("typed IR AST:\n{}\n".format(astor.dump_tree(lp_ast_type_checked)))
    def f(*args, **kwargs):
        backend_generator = frontend.BackendGenerate(*args, **kwargs)
        backend_generator.visit(lp_ast_type_checked)
        return backend_generator.remote_calls
    return f

def scope_lookup(var, scope):
    if (var in scope):
        return scope[var]
    elif "__parent__" in scope:
        return scope_lookup(var, scope["__parent__"])
    else:
        raise Exception(f"Scope lookup failed: scope={scope}, var={var}")

def eval_index_expr(index_expr, scope, dummify=False):
    bigm = scope_lookup(index_expr.matrix_name, scope)
    idxs = []
    for index in index_expr.indices:
        idxs.append(eval_expr(index, scope, dummify=dummify))
    return bigm, tuple(idxs)

def isinstance_fast(obj, typ):
    return type(obj) == typ

def eval_expr(expr, scope, dummify=False):
    if (isinstance(expr, sympy.Basic)):
        return expr
    elif (isinstance_fast(expr, int)):
        return expr
    elif (isinstance_fast(expr, float)):
        return expr
    if (isinstance_fast(expr, IntConst)):
        return expr.val
    elif (isinstance_fast(expr, FloatConst)):
        return expr.val
    elif (isinstance_fast(expr, BoolConst)):
        return expr.val
    elif (isinstance_fast(expr, str)):
        ref_val = scope_lookup(expr, scope)
        return eval_expr(ref_val, scope, dummify=dummify)
    elif (isinstance_fast(expr, Ref)):
        ref_val = scope_lookup(expr.name, scope)
        return eval_expr(ref_val, scope, dummify=dummify)
    elif (isinstance_fast(expr, BinOp)):
        left = eval_expr(expr.left, scope, dummify=dummify)
        right = eval_expr(expr.right, scope, dummify=dummify)
        return op_table[expr.op](left, right)
    elif (isinstance_fast(expr, CmpOp)):
        left = eval_expr(expr.left, scope, dummify=dummify)
        right = eval_expr(expr.right, scope, dummify=dummify)
        return op_table[expr.op](left, right)
    elif (isinstance_fast(expr, UnOp)):
        e = eval_expr(expr.e, scope, dummify=dummify)
        return op_table[expr.op](e)
    elif (isinstance_fast(expr, Mfunc)):
        e = eval_expr(expr.e, scope, dummify=dummify)
        return op_table[expr.op](e)
    elif (isinstance_fast(expr, RangeVar)):
        if (not dummify):
            raise Exception(f"Range variable {expr} cannot be evaluated directly, please specify a specific variable  or pass in dummify=True")
        else:
            return sympy.Symbol(expr.var)
    else:
        raise Exception(f"unsupported expr type {type(expr)}")


def eval_remote_call(r_call_with_scope, **kwargs):
    r_call = r_call_with_scope.remote_call
    compute = r_call.compute
    scope = copy_scope(r_call_with_scope.scope)
    scope.update(kwargs)
    pyarg_list = []
    pyarg_symbols = []
    for i, _arg in enumerate(r_call.args):
        pyarg_symbols.append(str(i))
        if (isinstance(_arg, IndexExpr)):
            matrix, indices = eval_index_expr(_arg, scope)
            arg = lp.RemoteRead(0, matrix, *indices)
        else:
            arg = eval_expr(_arg, scope)

        pyarg_list.append(arg)

    num_args = len(pyarg_list)
    num_outputs = len(r_call.output)
    if (r_call.kwargs is None):
        r_call_kwargs = {}
    else:
        r_call_kwargs = r_call.kwargs

    compute_instr  = lp.RemoteCall(0, compute, pyarg_list, num_outputs, pyarg_symbols, **r_call_kwargs)
    outputs = []

    for i, output in enumerate(r_call.output):
        assert isinstance(output, IndexExpr)
        matrix, indices = eval_index_expr(output, scope)
        op = lp.RemoteWrite(i + num_args, matrix, compute_instr.results[i], i, *indices)
        outputs.append(op)
    read_instrs =  [x for x in pyarg_list if isinstance(x, lp.RemoteRead)]
    write_instrs = outputs
    return lp.InstructionBlock(read_instrs + [compute_instr] + write_instrs)


def is_linear(expr, vars):
    if (expr.has(sympy.log)):
        return False

    if (expr.has(sympy.ceiling)):
        return False

    if (expr.has(sympy.floor)):
        return False

    if (expr.has(sympy.Pow)):
        return False
    return True
    '''
    for x in vars:
        for y in vars:
            try:
                if not sympy.Eq(sympy.diff(expr, x, y), 0):
                    return False
            except TypeError:
                return False
    return True
    '''

def is_constant(expr):
    if (isinstance(expr, Number)): return True
    return expr.is_constant()

def extract_vars(expr):
    if (isinstance(expr, Number)): return []
    return tuple(expr.free_symbols)

def extract_constant(expr):
    consts = [term for term in expr.args if not term.free_symbols]
    if (len(consts) == 0):
        return sympy.Integer(0)
    const = expr.func(*consts)
    return const

def _resort_var_names_by_limits(sol, solve_vars, limits):
    '''
        sol - dictionary
        var_names - list of strings
        limits - sympy lambda functions
    '''
    int_bounded_vars = {}
    const = False
    reg_vars = []
    for v in solve_vars:
        start,end,step = limits[v]
        start = start(**sol)
        end = end(**sol)
        step = step(**sol)
        if (is_constant(start) and is_constant(end) and is_constant(step)):
            assert isinstance(start, Number) or start.is_integer
            assert isinstance(end, Number) or end.is_integer
            assert isinstance(step, Number) or step.is_integer
            const = True
            int_bounded_vars[v] = float((end - start)/step)
        else:
            reg_vars.append(v)
    # if there are multiple bounded vars sort by least constrainted
    var_names_sorted_with_valid_range  = sorted(int_bounded_vars.keys(), key=lambda x: int_bounded_vars[x])
    var_names_sorted = var_names_sorted_with_valid_range + reg_vars
    return var_names_sorted, const





def symbolic_linsolve(A, b, solve_vars):
    eqs = []
    for a,b in zip(A,b):
        eqs.append(a - b)
    if np.all([x == 0 for x in eqs]):
        return []
    return list(sympy.linsolve(eqs, solve_vars))


def resplit_equations(A, C, b0, b1, solve_vars, sub_dict):
    ''' Split A, C into linear and non linear equations
        affter using sub_dict
    '''
    new_A = []
    new_C = []
    new_b0 = []
    new_b1 = []
    changed = False
    for eq, val in zip(A, b0):
        new_eq = eq.subs(sub_dict)
        new_A.append(new_eq)
        new_b0.append(val)
    for eq, val in zip(C, b1):
        new_eq = eq.subs(sub_dict)
        if (is_linear(new_eq, solve_vars)):
            changed = True
            new_A.append(new_eq)
            new_b0.append(val)
        else:
            new_C.append(new_eq)
            new_b1.append(val)
    assert len(new_C) <= len(C)
    return new_A, new_C, new_b0, new_b1, changed

def copy_scope(scope):
    new_scope = scope.copy()
    if "__parent__" in new_scope:
        new_scope["__parent__"] = scope["__parent__"].copy()
    return new_scope





def recursive_solver(A, C, b0, b1, solve_vars, var_limits, partial_sol):
    ''' Recursively solve set of linear + nonlinear equations
        @param A - is a list of sympy linear equations
        @param C - is a list of sympy nonlinear equations
        @param b0 - is a list of concrete values for linear equations
        @param b1 - is a list of concrete values for nonlinear equations
        @param solve_vars - is an ordered list of sympy variables currently solving for
        @param var_limits - dictionary from var_names to sympy functions expressing limits
    '''
    solutions = []
    assert set(solve_vars) == set(list(var_limits.keys()))
    if len(A) == 0:
        print("Warning: There are no linear equations to solve,\
              this will be a brute force enumeration")
        solution_candidate = None
    else:
        # A is nonempty
        x = symbolic_linsolve(A, b0, solve_vars)
        if (len(x) != 0):
            x = list(x)[0]
            sol_dict = dict(zip([str(x) for x in solve_vars], x))
            sol_dict.update(partial_sol)
            constant_sol = np.all([is_constant(var) for var in x])
            integral_sol = np.all([var.is_integer for var in x])
            if constant_sol and not integral_sol:
                # failure case 0 we get a constant fractional/decimal solution
                return []
            invalid_constants = np.all([is_constant(var) and (not var.is_integer) for var in x])
            if (invalid_constants):
                # failure case 1 we get a parametric solution with
                # fractional/decimal parts
                return []
            if (constant_sol):
                return [sol_dict]
            else:
                assert not constant_sol
                solve_vars, constant_range = _resort_var_names_by_limits(sol_dict, solve_vars, var_limits)
                # at least one variable should have constant range
                assert constant_range
                x = symbolic_linsolve(A, b0, solve_vars)
                assert(len(x) == 1)
                x = x[0]
                sol_dict = dict(zip([str(x) for x in solve_vars], x))
                sol_dict.update(partial_sol)
                solutions.append(sol_dict)
        else:
            return []
    if len(C) > 0:
        assert len(solutions) == 1

        A, C, b0, b1, changed = resplit_equations(A, C, b0, b1, solve_vars, solutions[0])

        if (changed):

            constant_vars = [v for v in solve_vars if is_constant(solutions[0][str(v)])]
            constant_sol = {}
            var_limits_recurse = var_limits.copy()
            solve_vars_recurse = solve_vars.copy()
            for v in constant_vars:
                del var_limits_recurse[v]
                solve_vars_recurse.remove(v)
                constant_sol[str(v)] = solutions[0][str(v)]
            res = recursive_solver(A, C, b0, b1, solve_vars_recurse, var_limits_recurse, constant_sol)
            [x.update(partial_sol) for x in res]
            return res
    assert len(solutions) == 1
    sol = solutions.pop(0)
    enumerate_var = solve_vars[0]
    start_f, end_f, step_f = var_limits[enumerate_var]
    t = time.time()
    start = start_f(**sol)
    e = time.time()
    end = end_f(**sol)
    step = step_f(**sol)
    for i in range(start, end, step):
        sol_i = sol.copy()
        sol_i[str(enumerate_var)] = sympy.Integer(i)
        A, C, b0, b1, changed  = resplit_equations(A, C, b0, b1, solve_vars, sol_i)
        constant_vars = [v for v in solve_vars if is_constant(sol_i[str(v)])]
        solve_vars_recurse = solve_vars.copy()
        var_limits_recurse = var_limits.copy()
        for v in constant_vars:
            del var_limits_recurse[v]
            solve_vars_recurse.remove(v)
        if (len(solve_vars_recurse) > 0):
            recurse_sols = recursive_solver(A, C, b0, b1, solve_vars_recurse, var_limits_recurse, sol_i)
            [x.update(sol_i) for x in recurse_sols]
            [x.update(partial_sol) for x in recurse_sols]
            solutions += recurse_sols
        else:
            sol_i.update(partial_sol)
            solutions.append(sol_i)

    return solutions


def prune_solutions(solutions, var_limits):
    valid_solutions = []
    for sol in solutions:
        bad_sol = False
        for k,v in sol.items():
            start,end,step = var_limits[sympy.Symbol(k)]
            start,end,step = start(**sol), end(**sol), step(**sol)
            if (v not in range(start, end, step)):
                bad_sol = True
        if (not bad_sol):
            valid_solutions.append(sol)
    return valid_solutions







def lambdify(expr):
    symbols = extract_vars(expr)
    _f = sympy.lambdify(symbols, expr, 'sympy', dummify=False)
    def f(**kwargs):
        if (len(kwargs) < len(symbols)):
            raise Exception("Insufficient Args")
        arg_dict = {}
        for s in symbols:
            arg_dict[str(s)] = kwargs.get(str(s), s)
        return _f(**arg_dict)
    return f




def template_match(page, offset, abstract_page, abstract_offset, offset_types, scope):
    ''' Look for possible template matches from abstract_page[abstract_offset] to page[offset] in scope '''
    # Form set of equations Z with abstract offset as LHS and offset as RHS
    # b = RHS
    # Partition Z into A and C where A is a list of *linear* equations and C is a list of *nonlinear* equations. b0 and b1 are the respective rhs.
    # call recursive_solver(A, C, b0, b1, var_names, var_limits):
        # if A is nonempty:
            # Solve x = A \ b0
            # if x is empty -> Return immediately
            # if x is integral -> Move on
            # if x is parametric ->
                # use partial solution to sort the columns of A such that the least
                # constrained variable is first
                # Resolve x = A \ b0
        # if C is nonempty:
            # new_C = []
            # new_A = []
            # new_b0 = []
            # new_b1 = []
            # for i in range(len(C))
                # res = C[i](x)
                # if (res is constant and res != b1):
                    # return []
                # if (res is linear):
                    # new_A.append(C[i])
                    # new_b0.append(b1[i])
                # else
                    # new_C.append(C[i])
                    # new_b1.append(b1[i])

            # if len(new_A) > 0:
                # assert len(new_A) + len(new_C) == len(C)
                # newAs, newBs = zip(*new_linear)
                # return solve([A; newAs], new_C, [b; new_b0], new_b1)
        # if we are here there were *no new* linear equations added so we can start enumerating
        # var, start, end, step = find_least_constrained_var(A, b0)
        # results = []
        # for i in range(start,end,step)
            # new_A = []
            # new_C = []
            # new_b0 = []
            # new_b1 = []
            # for eq, val in zip(A,b0):
                # new_eq = eq.subs({var: i})
                # new_A.append(new_eq)
                # new_B.append(val)
            # new_C = C.copy()
            # for eq, val in zip(C,b1):
                # new_eq = eq.subs({var: i})
                # if (is_linear(new_eq)):
                    # new_A.append(new_eq)
                    # new_b0.append(new_eq)
                # else:
                    # new_C.append(new_eq)
                    # new_b1.append(new_eq)
            # assert len(new_C) + len(new_A) == len(A) + len(C)
            # now arguments are ready for recursive call.
            # results += recursive_solver(new_A, new_C, new_b0, new_b1)

    assert abstract_page == page
    assert len(offset) == len(abstract_offset) == len(offset_types)
    vars_for_arg = list(set([z for x in abstract_offset for z in extract_vars(x)]))
    A = []
    b0 =[]
    C = []
    b1 = []
    for eq,val,offset_type in zip(abstract_offset, offset, offset_types):
        if issubclass(offset_type, LinearIntType):
            A.append(sympy.sympify(eq))
            b0.append(val)
        else:
            C.append(sympy.sympify(eq))
            b1.append(val)
    var_limits = {}
    var_limits_symbols = {}
    for var in vars_for_arg:
        range_var = scope_lookup(str(var), scope)
        assert (isinstance(range_var, RangeVar))
        start_val = eval_expr(range_var.start, scope, dummify=True)
        end_val = eval_expr(range_var.end, scope, dummify=True)
        step_val = eval_expr(range_var.step, scope, dummify=True)

        start_fn = lambdify(start_val)
        end_fn = lambdify(end_val)
        step_fn = lambdify(step_val)
        var_limits[var] = (start_fn, end_fn, step_fn)
        var_limits_symbols[var] = (start_val, end_val, step_val)
    sols = recursive_solver(A, C, b0, b1, vars_for_arg, var_limits, {})
    sols = prune_solutions(sols, var_limits)
    return sols



def find_parents(r_call, program, **kwargs):
    ''' Given a specific r_call and arguments to evaluate it completely
        return the program locations that writes to the input of r_call
    '''
    ib = eval_remote_call(r_call, **kwargs)
    parents = []
    for inst in ib.instrs:
        if (not isinstance(inst, lp.RemoteRead)): continue
        assert(isinstance(inst, lp.RemoteRead))
        page = inst.matrix
        offset = inst.bidxs
        for p_idx in program.keys():
            r_call_abstract_with_scope = program[p_idx]
            r_call_abstract = r_call_abstract_with_scope.remote_call
            scope = copy_scope(r_call_abstract_with_scope.scope)
            for i, output in enumerate(r_call_abstract.output):
                if (not isinstance(output, IndexExpr)): continue
                abstract_page, abstract_offset = eval_index_expr(output, scope, dummify=True)
                if (abstract_page != page): continue
                offset_types = [x.type for x in output.indices]
                local_parents = template_match(page, offset, abstract_page, abstract_offset, offset_types, scope)
                if (len(local_parents) > 1):
                    raise Exception("Invalid Program Graph, LambdaPackPrograms must be SSA")
                parents += [(p_idx, x) for x in local_parents]
    return utils.remove_duplicates(parents)

def find_children(r_call, program, **kwargs):
    ''' Given a specific r_call and arguments to evaluate it completely
        return all other program locations that read from the output of r_call
    '''
    ib = eval_remote_call(r_call, **kwargs)
    children = []
    for inst in ib.instrs:
        if (not isinstance(inst, lp.RemoteWrite)): continue
        assert(isinstance(inst, lp.RemoteWrite))
        page = inst.matrix
        offset = inst.bidxs
        for p_idx in program.keys():
            r_call_abstract_with_scope = program[p_idx]
            r_call_abstract = r_call_abstract_with_scope.remote_call
            scope = copy_scope(r_call_abstract_with_scope.scope)
            for i, arg in enumerate(r_call_abstract.args):
                if (not isinstance(arg, IndexExpr)): continue
                abstract_page, abstract_offset = eval_index_expr(arg, scope, dummify=True)
                if (abstract_page != page): continue
                offset_types = [x.type for x in arg.indices]
                local_children = template_match(page, offset, abstract_page, abstract_offset, offset_types, scope)
                children += [(p_idx, x) for x in local_children]
    return utils.remove_duplicates(children)

def extract_range_vars(scope):
    range_vars = {}
    for var, value in scope.items():
        if (type(value) == RangeVar):
            range_vars[var] = value
        if var == "__parent__":
            range_vars_parent = extract_range_vars(value)
            for k,v in range_vars_parent.items():
                if (k not in range_vars):
                    range_vars[k] = v
    return range_vars

def is_const_range_var(range_var, scope):
    start = range_var.start
    end = range_var.end
    step = range_var.step
    start = eval_expr(start, scope, dummify=True)
    end = eval_expr(end, scope, dummify=True)
    step = eval_expr(step, scope, dummify=True)
    return (type(start) == int) and (type(end) == int) and  (type(step) == int)

def delete_from_scope(scope, var):
    if var in scope:
        del scope[var]
    if "__parent__" in scope:
        delete_from_scope(scope["__parent__"], var)

def recursive_range_walk(scope):
    ''' Recursively walks scope and returns the set of all range vars '''
    range_vars = extract_range_vars(scope)
    const_range_vars = [k for (k,v) in range_vars.items() if is_const_range_var(v, scope)]
    if len(const_range_vars) == 0:
        return [{}]
    const_range_var = const_range_vars.pop(0)
    start = eval_expr(range_vars[const_range_var].start, scope)
    end = eval_expr(range_vars[const_range_var].end, scope)
    step = eval_expr(range_vars[const_range_var].step, scope)
    r_vals = []
    for i in range(start,end,step):
        scope_recurse = copy_scope(scope)
        delete_from_scope(scope_recurse, str(const_range_var))
        scope_recurse[str(const_range_var)] = i
        range_vars = extract_range_vars(scope_recurse)
        vals = recursive_range_walk(scope_recurse)
        [x.update({const_range_var: i}) for x in vals]
        r_vals += vals
    return r_vals







def walk_program(program):
    ''' Given a lambdapack program return all possible states
        this will be a large set for most reasonable programs use carefully!!
    '''
    states = []
    for p_idx in program.keys():
        r_call_abstract_with_scope = program[p_idx]
        scope = copy_scope(r_call_abstract_with_scope.scope)
        range_vars = recursive_range_walk(scope)
        states += [(p_idx, x) for x in range_vars]
    return states









