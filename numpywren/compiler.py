import ast
import astor
import inspect
import sympy
from numpywren.matrix import BigMatrix
from numpywren import exceptions, utils
from numpywren.lambdapack import RemoteRead, RemoteWrite, InstructionBlock, RemoteReturn, RemoteCall
from numpywren.matrix_utils import constant_zeros
from sympy.logic import boolalg
import dill
import logging
import numpy as np
import time

logger = logging.getLogger('numpywren')

class BackendStargs(ast.AST):
    _fields = ['args']

class BigMatrixBlock(object):
    def __init__(self, name, bigm, indices):
        self.name = name
        self.matrix = bigm
        self.indices = indices

    def __str__(self):
        read_str = ""
        read_str += self.matrix.key
        read_str += str(self.indices)
        return read_str

class Statement():
    def get_expr(self, index):
        pass

def scope_sub(expr, scope):
    if (isinstance(expr, sympy.Basic)):
        expr = expr.subs(scope)
        if "__parent__" in scope:
            return scope_sub(expr, scope["__parent__"])
        else:
            return expr
    elif (isinstance(expr, str)):
        return sympy.Symbol(expr)
    elif (isinstance(expr, int)):
        return sympy.Integer(expr)
    elif (isinstance(expr, float)):
        return sympy.Float(expr)
    else:
        return expr


class OperatorExpr(Statement):
    def __init__(self, opid, compute, args, outputs, scope, is_input=False, is_output=False, **options):
        ''' compute_expr is a subclass RemoteInstruction (not an instance)
             read_exprs, and write_exprs
            are of lists of type (BigMatrix, [sympy_expr, sympy_expr...])
        '''
        self.args  = args
        self.outputs = outputs
        self.compute = compute
        self.scope = scope
        self._read_exprs = [x for x in self.args if isinstance(x, BigMatrixBlock)]
        self._write_exprs = [x for x in self.outputs if isinstance(x, BigMatrixBlock)]
        self.opid = opid
        self.num_exprs = 1
        self.options = options
        self._is_input = is_input
        self._is_output = is_output


    def get_expr(self, index):
        assert index == 0
        return self

    def eval_big_matrix_block(self, bigm_block, var_values):
        idx_evaluated = ()
        for idx in bigm_block.indices:
            idx_subbed = scope_sub(idx, var_values)
            idx = scope_sub(idx_subbed, self.scope)
            if (idx.is_integer is None):
                raise Exception("Big matrix indices must evaluated to be integers, got {0} for {1}, var_values is {2}".format(idx, bigm_block, var_values))
            idx = int(idx)
            idx_evaluated += (idx,)
        return BigMatrixBlock(bigm_block.name, bigm_block.matrix, idx_evaluated)

    def eval_outputs(self, data_loc, var_values, num_args):
        outputs_evaluated = []
        for i, output in enumerate(self.outputs):
            if (isinstance(output, BigMatrixBlock)):
                block = self.eval_big_matrix_block(output, var_values)
                outputs_evaluated.append(RemoteWrite(i + num_args, block.matrix, data_loc, i, *block.indices))
            else:
                #TODO this is easily fixable
                raise Exception("Only BigMatrices can be outputs of RemoteCalls")
        return outputs_evaluated

    def eval_operator(self, var_values, **kwargs):
        if (isinstance(self.compute, RemoteReturn)):
            return InstructionBlock([self.compute])
        args_evaluated = self.eval_args(var_values)
        pyarg_list = []
        pyarg_symbols = []
        for i, arg in enumerate(args_evaluated):
            pyarg_symbols.append(str(i))
            if (isinstance(arg, RemoteRead)):
                pyarg_list.append(arg)

            elif (isinstance(arg, float)):
                pyarg_list.append(arg)
        num_outputs = len(self.outputs)
        compute_instr  = RemoteCall(0, self.compute, args_evaluated, num_outputs, pyarg_symbols, **kwargs)
        num_args = len(pyarg_list)
        write_instrs = self.eval_outputs(compute_instr.results, var_values, num_args)
        read_instrs =  [x for x in args_evaluated if isinstance(x, RemoteRead)]
        return InstructionBlock(read_instrs + [compute_instr] + write_instrs, priority=0)

    def eval_args(self, var_values):
        args_evaluated = []
        for i,arg in enumerate(self.args):
            if (isinstance(arg, BigMatrixBlock)):
                block = self.eval_big_matrix_block(arg, var_values)
                args_evaluated.append(RemoteRead(i, block.matrix, *block.indices))
            elif (isinstance(arg, sympy.Basic)):
                val = scope_sub(arg, self.scope)
                val = scope_sub(val, var_values)
                if (not val.is_constant()):
                    raise Exception("All values must be constant at eval time")
                args_evaluated.append(float(val))
            elif (isinstance(arg, BackendStargs) and arg.args.name == "__REDUCE_ARGS__"):
                reduction_args = self.eval_reduction_args(var_values)
                for r_arg in reduction_args:
                    for r in r_arg:
                        args_evaluated.append(RemoteRead(0, r.matrix, *r.indices))
            else:
               raise Exception("forbidden")
        return args_evaluated

    def find_reader_var_values(self, write_ref, var_names, var_limits, current_reduction_level=-1, conds=None):
        if (conds is None):
            conds = []

        results = []
        for read_expr in self._read_exprs:
            if write_ref.matrix != read_expr.matrix:
                continue
            if not var_names:
                if tuple(write_ref.indices) == tuple(read_expr.indices):
                    results.append({})
                continue
            assert len(write_ref.indices) == len(read_expr.indices)
            possibs = self._enumerate_possibilities(write_ref.indices, read_expr.indices,  var_names, var_limits, conds)
            results += possibs
        return utils.remove_duplicates(results)

    def find_writer_var_values(self, read_ref, var_names, var_limits, current_reduction_level=-1, conds=None):
        if (conds is None):
            conds = []
        if (len(self._write_exprs)) == 0:
            return []
        results = []
        for write_expr in self._write_exprs:
            if read_ref.matrix != write_expr.matrix:
                return []
            if not var_names:
                if tuple(read_ref.indices) == tuple(write_expr.indices):
                    return [{}]
            assert len(read_ref.indices) == len(write_expr.indices)
            results += self._enumerate_possibilities(read_ref.indices, write_expr.indices,
                                                    var_names, var_limits, conds=conds)
        return utils.remove_duplicates(results)

    def eval_read_refs(self, var_values):
        read_refs = []
        for i,arg in enumerate(self.args):
            if (isinstance(arg, BigMatrixBlock)):
                read_refs.append(self._eval_block_ref(arg, var_values))
            elif (isinstance(arg, sympy.Basic)):
                continue
            elif (isinstance(arg, BackendStargs) and arg.args.name == "__REDUCE_ARGS__"):
                reduction_args = self.eval_reduction_args(var_values)
                for matrix, idxs in reduction_args:
                    args_evaluated.append(BigMatrixBlock("REDUCTION_ARG", matrix, *idxs))
        return read_refs

    def eval_write_ref(self, var_values):
        write_refs = []
        for i, output in enumerate(self.outputs):
            if (isinstance(output, BigMatrixBlock)):
                write_refs.append(self._eval_block_ref(output, var_values))
            else:
                #TODO this is easily fixable
                raise Exception("Only BigMatrices can be outputs of RemoteCalls")
        return write_refs



    def _eval_block_ref(self, block_expr, var_values):
        return BigMatrixBlock(block_expr.name, block_expr.matrix, tuple([scope_sub(idx, self.scope).subs(var_values) for idx in block_expr.indices]))

    def _enumerate_possibilities(self, in_ref, out_expr, var_names, var_limits, conds):
        s = time.time()
        # pad loops with loop step if not explicitly provided
        for i,c in enumerate(var_limits):
            if(len(var_limits[i]) < 3):
                var_limits[i] += (1,)
            else:
                pass

        def check_cond(conds, var_values):
            ''' Check if any of conds evaluated to False
                returns (True, True) if all conds evaluated to True
                returns (True, False) if any of conds evaluated to False
                returns (False, None) if  no conds have evaluated to False and 
                not all conditions have been evaluated (i.e there are still free variables)
                '''

            resolved = 0
            for cond in conds:
                subbed = scope_sub(cond, self.scope).subs(var_values)
                if (isinstance(subbed, boolalg.BooleanAtom)):
                    resolved += 1
                    if (not bool(subbed)):
                        return (True, False)
            if (resolved == len(conds)):
                return (True, True)
            else:
                return (False, None)


        def brute_force_limits(sol, var_values, var_names, var_limits, conds):
            simple_var = None
            simple_var_idx = None
            var_limits = [(scope_sub(low, self.scope).subs(var_values), scope_sub(high, self.scope).subs(var_values), scope_sub(step, self.scope).subs(var_values)) for (low, high, step) in var_limits]
            for (i,(var_name, (low,high,step))) in enumerate(zip(var_names, var_limits)):
                low = low.subs(var_values)
                high = low.subs(var_values)
                if (not low.is_Symbol) and (not high.is_Symbol):
                    simple_var = var_name
                    simple_var_idx = i
                    break
            if (simple_var is None):
                raise Exception("NO simple var in loop")
            limits = var_limits[simple_var_idx]
            solutions = []
            if ((not sol[0].is_Symbol) and (sol[0] < limits[0] or sol[0] >= limits[1])):
                    return []
            simple_var_func = sympy.lambdify(simple_var, sol[0], "numpy")
            for val in range(limits[0], limits[1], limits[2]):
                var_values = var_values.copy()

                var_values[str(simple_var)] = int(simple_var_func(val))
                if(var_values[str(simple_var)] != val):
                    continue
                limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
                if (len(limits_left) > 0):
                    var_names_recurse = [x for x in var_names if x != simple_var]
                    solutions += brute_force_limits(sol[1:], var_values, var_names_recurse, limits_left, conds)
                else:
                    solutions.append(var_values)
            return solutions
        if (len(var_names) == 0): return []
        # brute force solve in_ref vs out_ref
        linear_eqns = [out_idx_expr - in_idx_ref for in_idx_ref, out_idx_expr in
                       zip(in_ref, out_expr)]
        #TODO sort var_names by least constrainted
        solve_vars  = []
        simple_var = None
        simple_var_idx = None
        var_limits = [(scope_sub(x, self.scope), scope_sub(y, self.scope), scope_sub(z, self.scope)) for (x,y,z) in var_limits]
        for (i,(var_name, (low,high,step))) in enumerate(zip(var_names, var_limits)):
            if (low.is_constant()) and (high.is_constant()):
                simple_var = var_name
                simple_var_idx = i
                break


        if (simple_var is None):
            raise Exception("NO simple var in loop: limits={0}, inref={1}, outref={2}".format(list(zip(var_names, var_limits)), in_ref, out_expr))

        solve_vars = [simple_var] + [x for x in var_names if x != simple_var]
        for i,x in enumerate(solve_vars):
            if (isinstance(x, str)):
                solve_vars[i] = sympy.Symbol(x)


        sols = list(sympy.linsolve(linear_eqns, solve_vars)) 
        # three cases
        # case 1 len(sols) == 0 -> no solution
        # case 2 exact (single) solution and solution is integer
        # case 3 parametric form -> enumerate
        if (len(sols) == 0):
            return []
        elif(len(sols) == 1):
            sol = sols[0]
            if (np.all([x.is_Symbol == False for x in sol]) and
                    np.all([x.is_Integer == True for x in sol])):
                solutions = [dict(zip([str(x) for x in solve_vars], sol))]
            else:
                limits = var_limits[simple_var_idx]
                solutions = []
                simple_var_func = sympy.lambdify(simple_var, sol[0], "numpy")
                if ((not sol[0].is_Symbol) and (sol[0] < limits[0] or sol[0] >= limits[1])):
                    return []
                for val in range(limits[0], limits[1], limits[2]):
                    var_values = {}
                    var_values[str(simple_var)] = int(simple_var_func(val))
                    if(var_values[str(simple_var)] != val):
                        continue
                    limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
                    if (len(limits_left) > 0):
                        var_names_recurse = [x for x in var_names if x != simple_var]
                        solutions += brute_force_limits(sol[1:], var_values, var_names_recurse, limits_left, conds)
                    else:
                        solutions.append(var_values)

        else:
            raise Exception("Linear equaitons should have 0, 1 or infinite solutions: {0}".format(sols))
        ret_solutions = []
        for sol in solutions:
            bad_sol = False
            resolved, val = check_cond(conds, sol)
            if (not resolved):
                raise Exception("Unresolved conditional conds={0}, var_values={1}".format(sol, conds))
            bad_sol = (bad_sol) or (not val)

            for (var_name, (low,high,step)) in zip(var_names, var_limits):
                low = low.subs(sol)
                high = high.subs(sol)
                if sol[str(var_name)] < low or sol[str(var_name)] >= high:
                    bad_sol = True

            if (not bad_sol):
                ret_solutions.append(sol)
        e = time.time()
        ret = utils.remove_duplicates(ret_solutions)
        return ret

    def __str__(self):
        read_str = ""
        for arg in self.args:
            if (isinstance(arg, BackendStargs)):
                read_str += "*"
                read_str += str(arg.args)
                read_str += ","
            elif (isinstance(arg, BigMatrixBlock)):
                read_str += arg.matrix.key
                read_str += str(arg.indices)
                read_str += ","
            elif (isinstance(arg, sympy.Basic)):
                read_str += str(arg)
                read_str += ","
            else:
                raise Exception("Unknown argument")
        read_str = read_str[:-1]
        if (len(self._write_exprs) > 0):
            write_str = ""
            for wexpr in self._write_exprs:
                write_str += wexpr.matrix.key
                write_str += str(wexpr.indices)
                write_str += ","
            str_expr = "{2} = {0}({1})".format(self.compute.__name__, read_str, write_str)
        else:
            str_expr = "{0}({1})".format(self.compute.__name__, read_str)

        if (self._is_input):
            str_expr = "__INPUT__ " + str_expr
        if (self._is_output):
            str_expr = "__OUTPUT__ " + str_expr
        return str(self.opid) + ":\t"  + str_expr

class ReductionOperatorExpr(OperatorExpr):
    def __init__(self, remote_call, var, limits, base_case, recursion, branch, scope, is_input=False, is_output=False, **options):
        self.remote_call = remote_call
        self.scope = scope
        self.branch = branch
        self.limits = limits
        self.base_case = base_case
        self.recursion = recursion
        self.is_input = is_input
        self.is_output = is_output
        self.var = var
        self.options = options
        self.num_exprs = 1
        self._is_input = is_input
        self._is_output = is_output
        self._read_exprs = remote_call._read_exprs
        self._write_exprs = remote_call._write_exprs
        self.outputs = remote_call.outputs
        self.args = remote_call.args
        self.compute = remote_call.compute
        self._is_input = remote_call._is_input
        self._is_output = remote_call._is_output


    def __str__(self):
        str_repr = "Reduction(base_case={base_case}, recursion={recursion}, branch={branch}, var={var}, limits={limits}, op={remote_call})"\
        .format(base_case=",".join([str(x) for x in self.base_case]), recursion=",".join([str(x) for x in self.recursion]), limits=self.limits, branch=self.branch, remote_call=self.remote_call, var=self.var)
        return str_repr

    def eval_reduction_args(self, var_values):
        reduction_level = var_values.get("__LEVEL__", 0)
        reduction_var = var_values[str(self.var)]
        reduction_branch = scope_sub(self.branch, var_values)
        reduction_branch = scope_sub(self.branch, self.scope)
        if (not reduction_branch.is_constant()):
            raise Exception("Reduction branch must be constant at eval time")
        reduction_branch = int(reduction_branch)
        start = scope_sub(self.limits[0], var_values)
        start = scope_sub(self.limits[0], self.scope)
        if (not start.is_constant()):
            raise Exception("Reduction start must be constant at eval time")
        start = int(start)

        end = scope_sub(self.limits[1], var_values)
        end = scope_sub(self.limits[1], self.scope)
        if (not end.is_constant()):
            raise Exception("Reduction end must be constant at eval time")

        end = int(end)
        if (reduction_var > end):
            raise Exception("Reduction var greater than reduction bound")

        if (reduction_var < 0):
            raise Exception("Reduction var must be non-negative")

        reduction_range = list(range(int(start), int(end), int(reduction_branch**reduction_level)))
        reduction_chunks = list(utils.chunk(reduction_range, reduction_branch))
        winning_chunk = None
        for i, chunk in enumerate(reduction_chunks):
            if (chunk[0] == reduction_var):
                winning_chunk = chunk
                break
        if (winning_chunk is None):
            raise Exception("Invalid reduction variable value")
        reduction_start = winning_chunk[0]
        reduction_end = winning_chunk[-1]
        if (reduction_level == 0):
            reduction_expr = self.base_case
        else:
            reduction_expr = self.recursion
        reduction_args = []
        for r_var in winning_chunk:
            var_values_reduce = var_values.copy()
            var_values_reduce[str(self.var)] = r_var
            var_values_reduce["__LEVEL__"] -= 1
            r_arg = ()
            for r_expr in reduction_expr:
                r_arg += (self.eval_big_matrix_block(r_expr, var_values_reduce),)
            reduction_args.append(r_arg)
        return reduction_args

    def eval_read_refs(self, var_values):
        read_refs = []
        for i,arg in enumerate(self.args):
            if (isinstance(arg, BigMatrixBlock)):
                read_refs.append(self._eval_block_ref(arg, var_values))
            elif (isinstance(arg, sympy.Basic)):
                continue
            elif (isinstance(arg, BackendStargs) and arg.args.name == "__REDUCE_ARGS__"):
                reduction_args = self.eval_reduction_args(var_values)
                for bigm_block_set in reduction_args:
                    for bigm_block in bigm_block_set:
                        read_refs.append(BigMatrixBlock("REDUCTION_ARG", bigm_block.matrix, bigm_block.indices))
        return read_refs

    def eval_write_ref(self, var_values):
        write_refs = []
        for i, output in enumerate(self.outputs):
            if (isinstance(output, BigMatrixBlock)):
                write_refs.append(self._eval_block_ref(output, var_values))
            else:
                #TODO this is easily fixable
                raise Exception("Only BigMatrices can be outputs of RemoteCalls")
        return write_refs



    def find_writer_var_values(self, read_ref, var_names, var_limits, current_reduction_level=-1, conds=None):
        if (conds is None):
            conds = []
        current_reduction_level -= 1
        if (current_reduction_level == -1):
            reduction_expr = self.base_case
        else:
            reduction_expr = self.recursion

        self.scope["__LEVEL__"] = current_reduction_level
        var_names.append(self.var)
        low, high = scope_sub(self.limits[0], self.scope), scope_sub(self.limits[1], self.scope)
        branch = scope_sub(self.branch, self.scope)
        var_limits.append((low, high, branch))
        results = []
        for output in self.outputs:
            if (output.matrix != read_ref.matrix):
                continue
            if not var_names:
                if tuple(read_ref.indices) == tuple(output.indices):
                    return [{}]
            assert len(read_ref.indices) == len(output.indices)
            idxs = [scope_sub(x, self.scope) for x in output.indices]
            results = self._enumerate_possibilities(read_ref.indices, idxs, var_names, var_limits, conds=conds)
        for r in results:
            r["__LEVEL__"] = current_reduction_level
        return utils.remove_duplicates(results)


    def find_reader_var_values(self, write_ref, var_names, var_limits, current_reduction_level=-1, conds=None):
        if (conds is  None):
            conds = []
        if (current_reduction_level == -1):
            reduction_expr = self.base_case
        else:
            reduction_expr = self.recursion
        var_names.append(self.var)
        low, high = scope_sub(self.limits[0], self.scope), scope_sub(self.limits[1], self.scope)
        branch = scope_sub(self.branch, self.scope)
        var_limits.append((low, high, branch))
        self.scope["__LEVEL__"] = current_reduction_level
        if ((current_reduction_level + 1) >= np.ceil(np.log(int(high))/np.log(int(branch)))):
            return []

        results = []
        for arg in self.remote_call.args:
            if (isinstance(arg, BigMatrixBlock)):
                if write_ref.matrix != arg.matrix:
                    continue
                if not var_names:
                    if tuple(write_ref.indices) == tuple(read_expr.indices):
                        results.append({})
                    continue
                assert len(write_ref.indices) == len(read_expr.indices)
                possibs = self._enumerate_possibilities(write_ref.indices, read_expr.indices,
                                                     var_names, var_limits, conds=conds)
                results += possibs
            elif (isinstance(arg, sympy.Basic)):
                continue
            elif (isinstance(arg, BackendStargs) and arg.args.name == "__REDUCE_ARGS__"):
                for reduction in reduction_expr:
                    idxs_subbed = [scope_sub(x, self.scope) for x in reduction.indices]
                    if write_ref.matrix != reduction.matrix:
                        continue
                    if not var_names:
                        if tuple(write_ref.indices) == tuple(reduction.indices):
                            results.append({})
                        continue
                    assert len(write_ref.indices) == len(reduction.indices)
                    possibs = self._enumerate_possibilities(write_ref.indices, idxs_subbed, var_names, var_limits, conds=conds)
                    results += possibs
        results = utils.remove_duplicates(results)
        branch = scope_sub(self.branch, self.scope)
        for r in results:
            r["__LEVEL__"] = (current_reduction_level + 1)
            r_var = r[str(self.var)]
            r_start = var_limits[-1][0]
            r_end = var_limits[-1][1]
            chunked_lst = list(utils.chunk(list(range(r_start, r_end, branch**(current_reduction_level+1))), branch))
            out_var = None
            for i,c in enumerate(chunked_lst):
                if (r_var in c):
                    out_var = i


            if (out_var == None):
                raise Exception("Invalid Reduction")
            r_var = chunked_lst[out_var][0]
            r[str(self.var)] = r_var
        return results

class BlockStatement(Statement):
    def __init__(self, body=[]):
        self._body = body
        self.num_stmts = 0
        self.num_exprs = 0
        for statement in self._body:
            self.num_exprs += statement.num_exprs
            self.num_stmts += 1

    def get_expr(self, index):
        assert index < self.num_exprs
        for statement in self._body:
            if index < statement.num_exprs:
                return statement.get_expr(index)
            index -= statement.num_exprs

    def add_statement(self, stmt):
      self._body.append(stmt)
      self.num_exprs += stmt.num_exprs
      self.num_stmts += 1

    @property
    def _is_output(self):
        outputs = False
        for stmt in self._body:
                outputs |= stmt._is_output
        return outputs

    @property
    def _is_input(self):
        inputs = False
        for stmt in self._body:
                inputs |= stmt._is_input
        return inputs






    def __str__(self):
        str_repr = ""
        for stmt in self._body:
            str_repr += str(stmt)
            str_repr += "\n"
        return str_repr


class Program(BlockStatement):
    def __init__(self, name, body, return_expr, symbols={}, all_symbols=set()):
        self.name = name
        self.return_expr = return_expr
        self.symbols = symbols
        self.all_symbols = all_symbols
        super().__init__(body)
        self.starters = self.find_starters()
        self.num_terminators = len(self.find_terminators())

    def get_children(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        reduction_level = -1
        if isinstance(operator_expr, ReductionOperatorExpr):
            reduction_level = var_values["__LEVEL__"]
            operator_expr.scope["__LEVEL__"] = reduction_level
        if (operator_expr == self.return_expr):
            return []
        write_refs = operator_expr.eval_write_ref(var_values)
        children = []
        for write_ref in write_refs:
            print("Looking for write ref ", write_ref)
            children += self.eval_read_operators(write_ref, current_reduction_level=reduction_level)
        if (operator_expr._is_output):
            children += [(self.num_exprs - 1, {})]

        return children

    def get_parents(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        if (operator_expr == self.return_expr):
            raise Exception("get_parents() shouldn't be called on return node, use self.num_terminators or find_terminators() instead")
        current_reduction_level  = -1
        if (isinstance(operator_expr, ReductionOperatorExpr)):
            current_reduction_level = var_values["__LEVEL__"]
            operator_expr.scope["__LEVEL__"] = current_reduction_level

        write_ref = operator_expr.eval_write_ref(var_values)
        parents = []
        operator_expr = self.get_expr(expr_idx)
        read_refs = operator_expr.eval_read_refs(var_values)

        for read_ref in read_refs:
            local_parents = self.eval_write_operators(read_ref, current_reduction_level=current_reduction_level)
            parents += local_parents
        parents = utils.remove_duplicates(parents)
        return parents

    def find_terminators(self):
        terminators = []
        expr_id = 0
        var_values = {}
        cached_sub_funcs = {}
        def recurse_find_terminators(body, expr_id, var_values, current_scope):
            for statement in body:
                if isinstance(statement, ReductionOperatorExpr):
                    if (not statement._is_output):
                        continue
                    else:
                        var = statement.var
                        sub_dict = {}
                        for s,v in self.all_symbols.items():
                            sub_dict[str(s)]  = v
                        for k,v in var_values.items():
                            sub_dict[str(k)] = v
                        if (statement in cached_sub_funcs):
                            min_idx_lambda, max_idx_lambda, branch_lambda = cached_sub_funcs[statement]
                        else:
                            symbols = [sympy.Symbol(x) for x in sub_dict.keys()]
                            min_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[0], current_scope), 'numpy', dummify=False)
                            branch_lambda = sympy.lambdify(symbols, scope_sub(statement.branch, current_scope), 'numpy', dummify=False)
                            max_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[1], current_scope), 'numpy', dummify=False)
                            cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda, branch_lambda)

                        start_val = min_idx_lambda(**sub_dict)
                        branch_val = branch_lambda(**sub_dict)
                        end_val = int(max_idx_lambda(**sub_dict))
                        num_reductions = max(int(np.ceil(np.log((end_val - start_val))/np.log(branch_val))), 1)

                        for i in range(num_reductions):
                            for j in range(start_val, end_val, branch_val**(i+1)):
                                var_values_recurse = var_values.copy()
                                var_values_recurse[str(var)] = j
                                var_values_recurse["__LEVEL__"] = i
                                terminators.append((expr_id, var_values_recurse))
                        expr_id += 1
                elif isinstance(statement, OperatorExpr):
                    if (not statement._is_output):
                        continue
                    else:
                        children =  self.get_children(expr_id, var_values)
                        terminators.append((expr_id, var_values))
                    expr_id += 1
                elif isinstance(statement, BackendIf):
                    if (not statement._is_output):
                        continue
                    cond_value = scope_sub(statement._cond, current_scope).subs(var_values)
                    if (not isinstance(cond_value, boolalg.BooleanAtom)):
                        raise Exception("ill specified conditional in if block: {0}".format(statement))
                    cond_value = bool(cond_value)
                    if (cond_value):
                        recurse_find_terminators(statement._if_body, expr_id, var_values, current_scope)
                        expr_id += statement.num_exprs
                    else:
                        expr_id += statement.num_exprs_if
                        recurse_find_terminators(statement._else_body, expr_id, var_values, current_scope)

                        expr_id += statement.num_exprs_else



                elif isinstance(statement, BackendFor):
                    var = statement.var
                    sub_dict = {}
                    for s,v in self.all_symbols.items():
                        sub_dict[str(s)]  = v
                    for k,v in var_values.items():
                        sub_dict[str(k)] = v
                    if (statement in cached_sub_funcs):
                        min_idx_lambda, max_idx_lambda = cached_sub_funcs[statement]
                    else:
                        symbols = [sympy.Symbol(x) for x in sub_dict.keys()]
                        min_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[0], current_scope), 'numpy', dummify=False)
                        max_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[1], current_scope), 'numpy', dummify=False)
                        cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda)
                    start_val = min_idx_lambda(**sub_dict)
                    end_val = int(np.ceil(max_idx_lambda(**sub_dict)))
                    outputs = False
                    # if there are no outputs in this for loop SKIP
                    for sub_statement in statement._body:
                            outputs |= sub_statement._is_output
                    if (outputs):
                        for z in range(start_val, end_val):
                            var_values_recurse = var_values.copy()
                            var_values_recurse[str(var)] = z
                            recurse_find_terminators(statement._body, expr_id, var_values_recurse, statement.symbol_table)
                    expr_id += statement.num_exprs

        recurse_find_terminators(self._body, expr_id, var_values, self.symbols)
        return utils.remove_duplicates(terminators)


    def find_starters(self):
        starters = []
        expr_id = 0
        var_values = {}
        cached_sub_funcs = {}
        def recurse_find_starters(body, expr_id, var_values, current_scope):
            for statement in body:
                if isinstance(statement, ReductionOperatorExpr):
                    if (not statement._is_input):
                        continue
                    else:
                        var = statement.var
                        sub_dict = {}
                        for s,v in self.all_symbols.items():
                            sub_dict[str(s)]  = v
                        for k,v in var_values.items():
                            sub_dict[str(k)] = v
                        if (statement in cached_sub_funcs):
                            min_idx_lambda, max_idx_lambda, branch_lambda = cached_sub_funcs[statement]
                        else:
                            symbols = [sympy.Symbol(x) for x in sub_dict.keys()]
                            min_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[0], current_scope), 'numpy', dummify=False)
                            branch_lambda = sympy.lambdify(symbols, scope_sub(statement.branch, current_scope), 'numpy', dummify=False)
                            max_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[1], current_scope), 'numpy', dummify=False)
                            cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda, branch_lambda)
                        start_val = min_idx_lambda(**sub_dict)
                        branch_val = branch_lambda(**sub_dict)
                        end_val = int(np.ceil(max_idx_lambda(**sub_dict)))
                        for j in range(start_val, end_val, branch_val):
                            var_values_recurse = var_values.copy()
                            var_values_recurse[str(var)] = j
                            var_values_recurse["__LEVEL__"] = 0
                            if (len(self.get_parents(expr_id, var_values_recurse)) == 0):
                                starters.append((expr_id, var_values_recurse))
                elif isinstance(statement, BackendIf):
                    if (not statement._is_input):
                        continue
                    cond_value = scope_sub(statement._cond, current_scope).subs(var_values)
                    if (not isinstance(cond_value, boolalg.BooleanAtom)):
                        raise Exception("ill specified conditional in if block: {0}".format(statement))
                    cond_value = bool(cond_value)
                    if (cond_value):
                        recurse_find_starters(statement._if_body, expr_id, var_values, current_scope)
                        expr_id += statement.num_exprs
                    else:
                        expr_id += statement.num_exprs_if
                        recurse_find_starters(statement._else_body, expr_id, var_values, current_scope)
                        expr_id += statement.num_exprs_else

                elif isinstance(statement, OperatorExpr):
                    if (not statement._is_input):
                        continue
                    elif (len(self.get_parents(expr_id, var_values)) == 0):
                        starters.append((expr_id, var_values))
                    else:
                        continue
                    expr_id += 1
                if isinstance(statement, BackendFor):
                    var = statement.var
                    sub_dict = {}
                    for s,v in self.all_symbols.items():
                        sub_dict[str(s)]  = v
                    for k,v in var_values.items():
                        sub_dict[str(k)] = v
                    if (statement in cached_sub_funcs):
                        min_idx_lambda, max_idx_lambda = cached_sub_funcs[statement]
                    else:
                        symbols = [sympy.Symbol(x) for x in sub_dict.keys()]
                        min_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[0], current_scope), 'numpy', dummify=False)
                        max_idx_lambda = sympy.lambdify(symbols, scope_sub(statement.limits[1], current_scope), 'numpy', dummify=False)
                        cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda)
                    start_val = min_idx_lambda(**sub_dict)
                    end_val = int(np.ceil(max_idx_lambda(**sub_dict)))
                    inputs = False
                    # if there are no inputs in this for loop SKIP
                    for sub_statement in statement._body:
                        inputs |= sub_statement._is_input

                    if (inputs):
                        for j in range(start_val, end_val):
                            var_values_recurse = var_values.copy()
                            var_values_recurse[str(var)] = j
                            recurse_find_starters(statement._body, expr_id, var_values_recurse, statement.symbol_table)
                    expr_id += statement.num_exprs

        recurse_find_starters(self._body, expr_id, var_values, self.symbols)
        return starters


    def eval_read_operators(self, write_ref, current_reduction_level=-1):
        def recurse_eval_read_ops(body, expr_counter, var_names, var_limits, conds):
            read_ops = []
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    valid_var_values = statement.find_reader_var_values(
                        write_ref, var_names, var_limits, current_reduction_level=current_reduction_level, conds=conds)
                    read_ops += [(expr_counter, vals) for vals in valid_var_values]
                elif isinstance(statement, BackendFor):
                    read_ops += recurse_eval_read_ops(statement._body,
                                                      expr_counter,
                                                      var_names + [str(statement.var)], 
                                                      var_limits + [statement.limits],
                                                      conds = [])
                elif isinstance(statement, BackendIf):

                    read_ops += recurse_eval_read_ops(statement._if_body,
                                                      expr_counter,
                                                      var_names,
                                                      var_limits,
                                                      conds + [statement._cond])
                    if (len(statement._else_body) > 0):
                        read_ops += recurse_eval_read_ops(statement._else_body,
                                                          expr_counter,
                                                          var_names,
                                                          var_limits,
                                                          conds + [~statement._cond])

                else:
                    raise NotImplementedError
                expr_counter += statement.num_exprs
            return read_ops
        return recurse_eval_read_ops(self._body, 0, [], [], [])

    def eval_write_operators(self, read_ref, current_reduction_level):
        def recurse_eval_write_ops(body, expr_counter, var_names, var_limits, conds):
            write_ops = []
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    valid_var_values = statement.find_writer_var_values(
                        read_ref, var_names, var_limits, current_reduction_level=current_reduction_level, conds=conds)
                    write_ops += [(expr_counter, vals) for vals in valid_var_values]
                elif isinstance(statement, BackendFor):
                    write_ops += recurse_eval_write_ops(statement._body,
                                                        expr_counter,
                                                        var_names + [statement.var],
                                                        var_limits + [statement.limits],
                                                        conds = conds)
                elif isinstance(statement, BackendIf):

                    write_ops += recurse_eval_write_ops(statement._if_body,
                                                      expr_counter,
                                                      var_names,
                                                      var_limits,
                                                      conds + [statement._cond])
                    if (len(statement._else_body) > 0):
                        write_ops += recurse_eval_write_ops(statement._else_body,
                                                          expr_counter,
                                                          var_names,
                                                          var_limits,
                                                          conds + [~statement._cond])

                else:
                    raise NotImplementedError
                expr_counter += statement.num_exprs
            return write_ops
        return recurse_eval_write_ops(self._body, 0, [], [], [])

    def __str__(self):
        str_repr = "def {0}():\n".format(self.name)
        for stmt in self._body:
            new_stmt_str = ["\t" + x for x in str(stmt).split("\n")]
            for substmt in new_stmt_str:
                str_repr += substmt + "\n"
        return str_repr


class BackendFor(BlockStatement):
    def __init__(self, var, limits, body, symbol_table):
        self.var = var
        self.limits = limits
        self.symbol_table = symbol_table
        super().__init__(body)

    def __str__(self):
        str_repr = "for {0} in range({1},{2}):".format(self.var, self.limits[0], self.limits[1])
        str_repr += "\n"
        for stmt in self._body:
            new_stmt_str = ["\t" + x for x in str(stmt).split("\n")]
            for substmt in new_stmt_str:
                str_repr += substmt + "\n"
        return str_repr

class BackendIf(BlockStatement):
    def __init__(self, cond, if_body, else_body):
        self.num_stmts = 0
        self.num_exprs = 0
        self.num_exprs_if = 0
        self.num_exprs_else = 0
        self._cond = cond
        self._if_body = if_body
        self._else_body = else_body
        self._body = self._if_body + self._else_body
        for statement in self._if_body:
            self.num_exprs += (statement.num_exprs)
            self.num_exprs_if += (statement.num_exprs)
            self.num_stmts += 1
        for statement in self._else_body:
            self.num_exprs += (statement.num_exprs)
            self.num_exprs_else += (statement.num_exprs)
            self.num_stmts += 1

    def __str__(self):
        str_repr = ""
        str_repr += "if {0}:".format(self._cond)
        str_repr += "\n"
        for x in self._if_body:
            str_repr += "\t" + str(x)
            str_repr += "\n"
        if (len(self._else_body) != 0):
            str_repr += "else:"
            str_repr += "\n"
            for x in self._else_body:
                str_repr += "\t" + str(x)
                str_repr += "\n"
        return str_repr




def make_3d_input_parent_fn(I):
    async def parent_fn_async(self, loop, *bidxs):
        if bidxs[0] == 0:
            return (await I.get_block_async(loop, *bidxs[1:]))[np.newaxis]
        else:
            exist = self.block_idxs_exist
            raise Exception("This shouldn't happen {0}, {1}".format(bidxs, exist))
            return constant_zeros(self, *bidxs)[np.newaxis]
    return parent_fn_async

