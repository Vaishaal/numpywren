import ast
import inspect
import astor
import sympy
from numpywren.matrix import BigMatrix
from numpywren import exceptions, utils
from numpywren.lambdapack import RemoteCholesky, RemoteTRSM, RemoteSYRK, RemoteRead, RemoteWrite, InstructionBlock, RemoteReturn, RemoteIdentity, RemoteGEMM, RemoteSUM
from numpywren.matrix_utils import constant_zeros
import numpy as np
import time
import dill
import logging

########
## ir ##
########
"""
expr = binop(bop op, expr left, expr right)
     | cmpop(cop op, expr left, expr right)
     | lpcall(lpop op, idxlist reads, idxlist writes)
     | unop(uop op, expr e)
     | ref(str name, expr? index)
     | intconst(int val)
     | idxlist(exprs* idxs)

uop = neg | not
bop = add | sub | mul | div | mod | and | or
cop =  eq |  ne |  lt |  gt |  le | ge
lpop = chol | syrk | trsm

stmt = assign(ref ref, expr val)
     | block(stmt* body)
     | if(expr cond, stmt body, stmt? elsebody)
     | for(str var, expr min, expr max, stmt body)
     | return(expr val)
	 | funcdef(str name, str* args, stmt body)
"""


# this is the frontend

VALID_CALLS = ['cholesky','syrk','trsm', 'identity', 'gemm']
VALID_REDUCE_OPS = ['sum']
COMPUTE_EXPR_MAP = {"cholesky": RemoteCholesky, "syrk": RemoteSYRK, "trsm":RemoteTRSM, "identity":RemoteIdentity, "gemm":RemoteGEMM}

VALID_TYPES = {'BigMatrix', 'int'}
logger = logging.getLogger(__name__)

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']

class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']

class UnOp(ast.AST):
    _fields = ['op', 'e']

class Ref(ast.AST):
    _fields = ['name']


class IntConst(ast.AST):
    _fields = ['val',]

class FloatConst(ast.AST):
    _fields = ['val',]

## Stmts ## class Assign(ast.AST):
    _fields = ['ref', 'val']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']

    def __init__(self, cond, body, elseBody=[]):
        return super().__init__(cond, body, elseBody)

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body', 'parallel']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']

class NumpywrenRead(ast.AST):
    _fields = ['matrix_block']

class NumpywrenWrite(ast.AST):
    _fields = ['matrix_block']

class BigMatrixBlock(ast.AST):
    _fields = ['name', 'bigm', 'bidx']

class Lambdapack(ast.AST):
    _fields = ['compute', 'reads', 'writes', 'options', 'is_input', 'is_output']

class Reduce(ast.AST):
    _fields = ['op', 'var', 'start', 'end', 'expr']



class NumpywrenParse(ast.NodeVisitor):
    """
    Translate a lambdapack expression.
    """
    def __init__(self, args, kwargs, inputs, outputs):
        super().__init__()
        self.input_args = args
        self.input_kwargs =  kwargs
        self.inputs = set(inputs)
        self.outputs = set(outputs)
        self.symbols = {}
        self.symbol_types = {}
        self.loop_variables = set()

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        elif isinstance(node.n, float):
            return FloatConst(val=node.n)
        else:
            raise NotImplementedError("Only Integers and Floats supported")

    def visit_BinOp(self, node):
        VALID_BOPS  = ["Add", "Sub", "Mult", "Div", "Mod", "And", "Or"]
        left = self.visit(node.left)
        right = self.visit(node.right)
        op  = node.op.__class__.__name__
        if (op not in VALID_BOPS):
            raise NotImplementedError("Unsupported BinOp {0}".format(op))
        return BinOp(op, left, right)

    def visit_Str(self, node):
        return node.s

    def visit_Compare(self, node):
        VALID_COMP_OPS = ["EQ", "NE", "LT", "GT",  "LE", "GE"]
        left = self.visit(node.left)
        if (len(node.ops) != 1):
            raise NotImplementedError("Only single op compares supported")

        if (len(node.comparators) != 1):
            raise NotImplementedError("Only single comparator compares supported")

        right = self.visit(node.comparators[0])

        op  = node.ops[0].__class__.__name__
        s_op = op.upper()
        if (s_op not in VALID_COMP_OPS):
            raise NotImplementedError("Unsupported CmpOp {0}".format(s_op))
        return CmpOp(s_op, left, right)

    def visit_UnaryOp(self, node):
        e = self.visit(node.operand)

        op = node.op.__class__.__name__
        if (op == "USub"):
            s_op = "Neg"
        elif (op == "Not"):
            s_op = "Not"
        else:
            raise NotImplementedError("Unsupported unary operation {0}".format(op))
        return UnOp(s_op, e)


    def visit_Name(self, node):
        if (node.id in self.symbols):
            val = self.symbols[node.id]
            if (isinstance(val, int)):
                return IntConst(val)
            elif (isinstance(val, float)):
                return FloatConst(val)

        return Ref(node.id)

    def visit_NameConstant(self, node):
        if (node.value == True):
            return IntConst(1)
        elif (node.value == False):
            return IntConst(0)
        else:
            raise exceptions.LambdaPackParsingException("Unsupported Name constant")


    def visit_Assign(self, node):
        if (len(node.targets) != 1):
            raise NotImplementedError("Only single argument assignment supported")

        options = {}
        writes = [self.visit(node.targets[0])]
        if (isinstance(node.value, ast.Call)):
            if (node.value.func.id == "reduce"):
                kwargs = {x.arg: self.visit(x.value) for x in node.value.keywords}
                assert isinstance(kwargs["var"], Ref)
                assert isinstance(kwargs["expr"], BigMatrixBlock)
                assert kwargs["op"] in VALID_REDUCE_OPS, "Invalid reduction operation"
                return Reduce(**kwargs)

            assert(isinstance(node.targets[0], ast.Subscript))
            print(node.value.func.id)
            assert(node.value.func.id in VALID_CALLS)
            reads = []
            for arg in node.value.args:
                assert(isinstance(arg, ast.Subscript) or isinstance(arg, ast.Attribute))
                reads.append(self.visit(arg))
            func_id = node.value.func.id
            for kw in node.value.keywords:
                options[kw.arg] = self.visit(kw.value)
        elif (isinstance(node.value, ast.Subscript) or isinstance(node.value, ast.Attribute)):
            reads = [self.visit(node.value)]
            writes = [self.visit(node.targets[0])]
            func_id = "identity"

        if (set([read.name for read in reads]).issubset(self.inputs)):
            is_input = True
        else:
            is_input = False

        if (len(set([write.name for write in writes]).intersection(self.outputs)) > 0):
            is_output = True
        else:
            is_output = False
        return Lambdapack(func_id, reads, writes, options, is_input, is_output)





    def visit_If(self, node):
        cond = self.visit(node.test)
        body = [self.visit(x) for x in node.body]
        else_body = [self.visit(x) for x in node.orelse]
        return If(cond, body, else_body)

    def visit_FunctionDef(self, func):
        args = [x.arg for x in func.args.args]
        if (len(args) != len(self.input_args) + len(self.input_kwargs)):
            raise exceptions.LambdaPackParsingException("Incorrect number of function arguments")
        annotations = [x.annotation for x in func.args.args]
        if (None in annotations):
            raise exceptions.LambdaPackParsingException("All function arguments must typed")
        annotations = [x.id for x in annotations]

        if (set(annotations) != VALID_TYPES):
            raise exceptions.LambdaPackParsingException("All function arguments must be of types: {0}".format(VALID_TYPES))
        for arg_name, arg in zip(args,self.input_args):
            self.symbols[arg_name] = arg
        for arg_name, arg in self.input_kwargs.items():
            if arg_name not in args:
                raise exceptions.LambdaPackParsingException("Unexpected argument {0}".format(arg_name))
            self.symbols[arg_name] = arg

        for arg_name, arg_type in zip(args, annotations):
            self.symbol_types[arg_name] = arg_type


        name = func.name
        assert isinstance(func.body, list)
        body = [self.visit(x) for x in func.body]
        return FuncDef(func.name, args, body)


    def visit_For(self, node):
        iter_node = node.iter
        is_call = isinstance(iter_node, ast.Call)
        if (is_call):
            is_range = iter_node.func.id == "range"
            is_par_range = iter_node.func.id == "par_range"
        else:
            is_range = False

        if (not is_call):
            raise NotImplementedError("Only for(x in range(...)) loops allowed")

        if (not (is_range or is_par_range)):
            raise NotImplementedError("Only for(x in range(...)) loops allowed")


        if(len(iter_node.args) > 2):
            raise NotImplementedError("Only 1-2 argument ranges allowed in for loop")

        if (len(iter_node.args) == 1):
            start = IntConst(0)
            end = self.visit(iter_node.args[0])
        else:
            start = self.visit(iter_node.args[0])
            end = self.visit(iter_node.args[1])

        body = [self.visit(x) for x in node.body]
        var = node.target.id
        self.loop_variables.add(node.target.id)
        return For(var, start, end, body, is_par_range)

    def visit_Attribute(self, node):
        assert(node.attr == 'T')
        child = self.visit(node.value)
        assert(isinstance(child, BigMatrixBlock))
        child.bigm = child.bigm.T
        return child

    def visit_Subscript(self, node):
        if (isinstance(node.slice.value, ast.Tuple)):
            if (node.value.id not in self.symbols):
                raise exceptions.LambdaPackParsingException("Unknown BigMatrix references")
            return BigMatrixBlock(node.value.id, self.symbols[node.value.id], [self.visit(x) for x in node.slice.value.elts])
        else:
            val = self.visit(node.slice.value) 
            assert(isinstance(val, IntConst) or isinstance(val, Ref))
            return BigMatrixBlock(node.value.id, self.symbols[node.value.id], [val])
    def visit_Return(self, node):
        assert False, "Returns are not permitted in lambdapack"







def lpcompile(function, inputs, outputs):
    def func(*args, **kwargs):
        logger.warning("Source code : \n{}\n".format(inspect.getsource(function)))
        function_ast = ast.parse(inspect.getsource(function)).body[0]
        print("Python AST:\n{}\n".format(astor.dump(function_ast)))
        parser = NumpywrenParse(args, kwargs, inputs, outputs)
        lp_ast = parser.visit(function_ast)
        logger.warning("IR AST:\n{}\n".format(astor.dump_tree(lp_ast)))
        program = BackendGenerator.generate_program(lp_ast)
        logger.warning("Program: {0}".format(program))
        return program
    return func


class Statement():
    def get_expr(self, index):
        pass


class OperatorExpr(Statement):
    def __init__(self, opid, compute_expr, read_exprs, write_expr, is_input=False, is_output=False, **options):
        ''' compute_expr is a subclass RemoteInstruction (not an instance)
             read_exprs, and write_exprs
            are of lists of type (BigMatrix, [sympy_expr, sympy_expr...])
        '''
        self._read_exprs = read_exprs
        self._write_expr = write_expr
        self.compute_expr = compute_expr
        self.opid = opid
        self.num_exprs = 1
        self.options = options
        self._is_input = is_input
        self._is_output = is_output
        for bigm, expr in read_exprs:
            expr



    def get_expr(self, index):
        assert index == 0
        return self

    def eval_operator(self, var_values, **kwargs):
        read_instrs = self.eval_read_instrs(var_values)
        compute_instr = self._eval_compute_instr(read_instrs, **kwargs)
        if (self._write_expr != []):
            write_instr = [self.eval_write_instr(compute_instr, var_values)]
        else:
            write_instr = []
        return InstructionBlock(read_instrs + [compute_instr] + write_instr, priority=0)

    def _eval_compute_instr(self, read_instrs, **kwargs):
        options = dict(list(self.options.items()) + list(kwargs.items()))
        return self.compute_expr(self.opid, read_instrs, **options)

    def find_reader_var_values(self, write_ref, var_names, var_limits):
        results = []
        for read_expr in self._read_exprs:
            if write_ref[0] != read_expr[0]:
                continue
            if not var_names:
                if tuple(write_ref[1]) == tuple(read_expr[1]):
                    results.append({})
                continue
            assert len(write_ref[1]) == len(read_expr[1])
            possibs = self._enumerate_possibilities(write_ref[1], read_expr[1],
                                                     var_names, var_limits)
            results += possibs
        return utils.remove_duplicates(results)

    def find_writer_var_values(self, read_ref, var_names, var_limits):
        if (len(self._write_expr)) == 0:
            return []
        if read_ref[0] != self._write_expr[0]:
            return []
        if not var_names:
            if tuple(read_ref[1]) == tuple(self._write_expr[1]):
                return [{}]
        assert len(read_ref[1]) == len(self._write_expr[1])
        results = self._enumerate_possibilities(read_ref[1], self._write_expr[1],
                                                var_names, var_limits)
        return utils.remove_duplicates(results)

    def eval_read_instrs(self, var_values):
        read_refs = self.eval_read_refs(var_values)
        return [RemoteRead(0, ref[0], *ref[1]) for ref in read_refs]

    def eval_write_instr(self, data_instr, var_values):
        write_ref = self.eval_write_ref(var_values)
        return RemoteWrite(0, write_ref[0], data_instr, *write_ref[1])

    def eval_read_refs(self, var_values):
        return [self._eval_block_ref(block_expr, var_values) for block_expr in self._read_exprs]

    def eval_write_ref(self, var_values):
        return self._eval_block_ref(self._write_expr, var_values)

    def _eval_block_ref(self, block_expr, var_values):
        return (block_expr[0], tuple([idx.subs(var_values) for idx in block_expr[1]]))

    def _enumerate_possibilities(self, in_ref, out_expr, var_names, var_limits):
        s = time.time()
        def brute_force_limits(sol, var_values, var_names, var_limits):
            simple_var = None
            simple_var_idx = None
            var_limits = [(low.subs(var_values), high.subs(var_values)) for (low, high) in var_limits]
            for (i,(var_name, (low,high))) in enumerate(zip(var_names, var_limits)):
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
            for val in range(limits[0], limits[1]):
                var_values = var_values.copy()

                var_values[simple_var] = int(simple_var_func(val))
                if(var_values[simple_var] != val):
                    continue
                limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
                if (len(limits_left) > 0):
                    var_names_recurse = [x for x in var_names if x != simple_var]
                    solutions += brute_force_limits(sol[1:], var_values, var_names_recurse, limits_left)
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
        for (i,(var_name, (low,high))) in enumerate(zip(var_names, var_limits)):
            if (not low.is_Symbol) and (not high.is_Symbol):
                simple_var = var_name
                simple_var_idx = i
                break
        if (simple_var is None):
            raise Exception("NO simple var in loop: limits={0}, inref={1}, outref={2}".format(list(zip(var_names, var_limits)), in_ref, out_expr))
        solve_vars = [simple_var] + [x for x in var_names if x != simple_var]
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
                solutions = [dict(zip(solve_vars, sol))]
            else:
                limits = var_limits[simple_var_idx]
                solutions = []
                simple_var_func = sympy.lambdify(simple_var, sol[0], "numpy")
                if ((not sol[0].is_Symbol) and (sol[0] < limits[0] or sol[0] >= limits[1])):
                    return []
                for val in range(limits[0], limits[1]):
                    var_values = {}
                    var_values[simple_var] = int(simple_var_func(val))
                    if(var_values[simple_var] != val):
                        continue
                    limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
                    if (len(limits_left) > 0):
                        var_names_recurse = [x for x in var_names if x != simple_var]
                        solutions += brute_force_limits(sol[1:], var_values, var_names_recurse, limits_left)
                    else:
                        solutions.append(var_values)

        else:
            raise Exception("Linear equaitons should have 0, 1 or infinite solutions: {0}".format(sols))
        ret_solutions = []
        for sol in solutions:
            bad_sol = False
            for (var_name, (low,high)) in zip(var_names, var_limits):
                low = low.subs(sol)
                high = high.subs(sol)
                if sol[var_name] < low or sol[var_name] >= high:
                    bad_sol = True
            if (not bad_sol):
                ret_solutions.append(sol)
        e = time.time()
        ret = utils.remove_duplicates(ret_solutions)
        #print("Enumerate possibilities called for in_ref: {0}, out_expr: {1}, var_names: {2}, var_limits {3} and took {4} seconds, sols: {5}, simple_var_idx: {6}".format(in_ref, out_expr, var_names, var_limits, e - s, sols, simple_var_idx))
        return ret

    def _enumerate_possibilities_deprc(self, in_ref, out_expr, var_names, var_limits):
        def recurse_enum_pos(idx, var_limits, parametric_eqns):
            print("RECURSE", parametric_eqns)
            print("limits", var_limits)
            if not parametric_eqns:
                return []
            if len(parametric_eqns) == len(var_names):
                print("in if statement")
                var_values = {}
                value = []
                #print("limits",var_limits)
                new_limits = [(lim[0].subs(parametric_eqns), lim[1].subs(parametric_eqns))
                               for lim in var_limits]
                print("new limits", new_limits)
                print(var_names)
                for var, limit in zip(var_names, new_limits):
                    value = parametric_eqns[var]
                    print(value)
                    #if not value.is_Integer:
                    #    return []
                    #assert parametric_eqns[var].is_Integer
                    #assert limit[0].is_Integer
                    #assert limit[1].is_Integer
                    if parametric_eqns[var] < limit[0] or parametric_eqns[var] >= limit[1]:
                        return []
                    var_values[var] = int(value)
                return [var_values]

            curr_eqn = parametric_eqns.get(var_names[idx])
            curr_limit = var_limits[idx]
            print("Curr eqn", curr_eqn)
            if curr_eqn is None:
                refs = []
                print("in other if statement")
                print(curr_limit)
                for i in range(curr_limit[0], curr_limit[1]):
                    new_eqns = {name: eqn.subs({var_names[idx]: i})
                                for name, eqn in parametric_eqns.items()}
                    new_eqns[var_names[idx]] = sympy.Integer(i)
                    new_limits = [(lim[0].subs(new_eqns), lim[1].subs(new_eqns))
                                  for lim in var_limits]
                    refs += recurse_enum_pos(idx + 1, new_limits, new_eqns) 
                return refs
            elif curr_eqn.is_integer:
                new_eqns = {name: eqn.subs({var_names[idx]: curr_eqn})
                            for name, eqn in parametric_eqns.items()}
                new_limits = [(lim[0].subs(new_eqns), lim[1].subs(new_eqns))
                              for lim in var_limits]
                return recurse_enum_pos(idx + 1, new_limits, new_eqns)
            else:
                return []
        linear_eqns = [out_idx_expr - in_idx_ref for in_idx_ref, out_idx_expr in
                       zip(in_ref, out_expr)]
        parametric_eqns = sympy.solve(linear_eqns, list(reversed(var_names)))
        print("LIMITS",var_limits)
        print(parametric_eqns)
        return recurse_enum_pos(0, var_limits, parametric_eqns)
    def __str__(self):
        read_str = ""
        for read_expr in self._read_exprs:
            read_str += read_expr[0].key
            read_str += str(read_expr[1])
            read_str += ","
        read_str = read_str[:-1]
        if (len(self._write_expr) > 0):
            write_str = ""
            write_str += self._write_expr[0].key
            write_str += str(self._write_expr[1])
            write_str += ","
            write_str = write_str[:-1]

            str_expr = "{2} = {0}({1})".format(self.compute_expr.__name__, read_str, write_str)
        else:
            str_expr = "{0}({1})".format(self.compute_expr.__name__, read_str)

        if (self._is_input):
            str_expr = "__INPUT__ " + str_expr
        if (self._is_output):
            str_expr = "__OUTPUT__ " + str_expr
        return str_expr



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
    def __init__(self, name, body, symbols={}):
        self.name = name
        self.symbols = symbols
        super().__init__(body)
        self.starters = self.find_starters()
        self.num_terminators = len(self.find_terminators())

    def get_children(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        if (operator_expr == self.return_expr):
            return []
        write_ref = operator_expr.eval_write_ref(var_values)
        children = self.eval_read_operators(write_ref)
        if (operator_expr._is_output):
            children += [(self.num_exprs - 1, {})]
        return children

    def get_parents(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        if (operator_expr == self.return_expr):
            raise Exception("get_parents() shouldn't be called on return node, use self.num_terminators or find_terminators() instead")
        write_ref = operator_expr.eval_write_ref(var_values)
        parents = []
        operator_expr = self.get_expr(expr_idx)
        read_refs = operator_expr.eval_read_refs(var_values)
        for read_ref in read_refs:
            local_parents = self.eval_write_operators(read_ref)
            parents += local_parents
        parents = utils.remove_duplicates(parents)
        return parents

    def find_terminators(self):
        terminators = []
        expr_id = 0
        var_values = {}
        cached_sub_funcs = {}
        def recurse_find_terminators(body, expr_id, var_values):
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    if (not statement._is_output):
                        continue
                    else:
                        terminators.append((expr_id, var_values))
                    expr_id += 1
                elif isinstance(statement, BackendFor):
                    symbols = tuple(self.symbols.keys())
                    sub_dict = self.symbols
                    for k,v in var_values.items():
                        sub_dict[str(k)] = v
                    if (statement in cached_sub_funcs):
                        min_idx_lambda, max_idx_lambda = cached_sub_funcs[statement]
                    else:
                        min_idx_lambda = sympy.lambdify(symbols, statement._limits[0], 'numpy', dummify=False)
                        max_idx_lambda = sympy.lambdify(symbols, statement._limits[1], 'numpy', dummify=False)
                        cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda)

                    start_val = min_idx_lambda(**sub_dict)
                    end_val = max_idx_lambda(**sub_dict)

                    #print("STARTING FOR LOOP FROM {0} to {1}".format(start_val, end_val))
                    var = statement._var
                    outputs = False
                    # if there are no outputs in this for loop SKIP
                    for sub_statement in statement._body:
                            outputs |= sub_statement._is_output
                    if (outputs):
                        for z in range(start_val, end_val):
                            var_values_recurse = var_values.copy()
                            var_values_recurse[var] = z
                            recurse_find_terminators(statement._body, expr_id, var_values_recurse)
                    expr_id += statement.num_exprs

        recurse_find_terminators(self._body, expr_id, var_values)
        return terminators

    def unroll_program(self):
        ''' Warning: this is very expensive! only use on small matrics and/or large block size, primarily for debugging purposes '''
        nodes  = []
        expr_id = 0
        var_values = {}
        cached_sub_funcs = {}
        def recurse_find_nodes(body, expr_id, var_values):
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    nodes.append((expr_id, var_values))
                    expr_id += 1
                if isinstance(statement, BackendFor):
                    symbols = tuple(self.symbols.keys())
                    sub_dict = self.symbols
                    for k,v in var_values.items():
                        sub_dict[str(k)] = v
                    if (statement in cached_sub_funcs):
                        min_idx_lambda, max_idx_lambda = cached_sub_funcs[statement]
                    else:
                        min_idx_lambda = sympy.lambdify(symbols, statement._limits[0], 'numpy', dummify=False)
                        max_idx_lambda = sympy.lambdify(symbols, statement._limits[1], 'numpy', dummify=False)
                        cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda)

                    start_val = min_idx_lambda(**sub_dict)
                    end_val = max_idx_lambda(**sub_dict)
                    var = statement._var
                    inputs = False
                    for j in range(start_val, end_val):
                        var_values_recurse = var_values.copy()
                        var_values_recurse[str(var)] = j
                        recurse_find_nodes(statement._body, expr_id, var_values_recurse)
                    expr_id += statement.num_exprs

        recurse_find_nodes(self._body, expr_id, var_values)
        return utils.remove_duplicates(nodes)

    def find_starters(self):
        starters = []
        expr_id = 0
        var_values = {}
        cached_sub_funcs = {}
        def recurse_find_starters(body, expr_id, var_values):
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    if (not statement._is_input):
                        continue
                    else:
                        starters.append((expr_id, var_values))
                    expr_id += 1
                if isinstance(statement, BackendFor):
                    symbols = tuple(self.symbols.keys())
                    sub_dict = self.symbols
                    for k,v in var_values.items():
                        sub_dict[str(k)] = v
                    if (statement in cached_sub_funcs):
                        min_idx_lambda, max_idx_lambda = cached_sub_funcs[statement]
                    else:
                        min_idx_lambda = sympy.lambdify(symbols, statement._limits[0], 'numpy', dummify=False)
                        max_idx_lambda = sympy.lambdify(symbols, statement._limits[1], 'numpy', dummify=False)
                        cached_sub_funcs[statement] = (min_idx_lambda, max_idx_lambda)

                    start_val = min_idx_lambda(**sub_dict)
                    end_val = max_idx_lambda(**sub_dict)
                    var = statement._var
                    inputs = False
                    # if there are no inputs in this for loop SKIP
                    for sub_statement in statement._body:
                        inputs |= sub_statement._is_input
                    if (inputs):
                        for j in range(start_val, end_val):
                            var_values_recurse = var_values.copy()
                            var_values_recurse[var] = j
                            recurse_find_starters(statement._body, expr_id, var_values_recurse)
                    expr_id += statement.num_exprs

        recurse_find_starters(self._body, expr_id, var_values)
        return starters

    def find_num_terminators(self):
        terminators = 0
        writes = {}
        opid = 0
        expr_count = 0
        var_values = {}
        def recurse_find_loop_length(for_statement, loop_var, start_val, end_val, var_values):
            tot_loop_length = 0
            for statement in for_statement._body:
                if isinstance(statement, BackendFor):
                    try:
                        sub_end_val = int(statement._limits[1].subs(var_values))
                        sub_start_val = int(statement._limits[0].subs(var_values))
                        local_loop_length = max(sub_end_val - sub_start_val, 0)
                        sub_loop_length = recurse_find_loop_length(statement, statement._var, sub_start_val, sub_end_val, values)
                        tot_loop_length += sub_loop_length
                    except:
                        # loop bound not fully specified so we enumerate
                        # TODO we can probably be smarter here
                        for i in range(start_val, end_val):
                            values = var_values.copy()
                            values[loop_var] = i
                            sub_end_val = int(statement._limits[1].subs(values))
                            sub_start_val = int(statement._limits[0].subs(values))
                            local_loop_length = max(sub_end_val - sub_start_val, 0)
                            sub_loop_length = recurse_find_loop_length(statement, statement._var, sub_start_val, sub_end_val, values)
                            tot_loop_length += sub_loop_length
                elif isinstance(statement, OperatorExpr):
                    tot_loop_length += end_val - start_val 
            return tot_loop_length


    def eval_read_operators(self, write_ref):
        def recurse_eval_read_ops(body, expr_counter, var_names, var_limits):
            read_ops = []
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    valid_var_values = statement.find_reader_var_values(
                        write_ref, var_names, var_limits)
                    read_ops += [(expr_counter, vals) for vals in valid_var_values]
                elif isinstance(statement, BackendFor):
                    read_ops += recurse_eval_read_ops(statement._body,
                                                      expr_counter,
                                                      var_names + [statement._var],
                                                      var_limits + [statement._limits])
                else:
                    raise NotImplementedError
                expr_counter += statement.num_exprs
            return read_ops
        return recurse_eval_read_ops(self._body, 0, [], [])

    def eval_write_operators(self, read_ref):
        def recurse_eval_write_ops(body, expr_counter, var_names, var_limits):
            write_ops = []
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    valid_var_values = statement.find_writer_var_values(
                        read_ref, var_names, var_limits)
                    write_ops += [(expr_counter, vals) for vals in valid_var_values]
                elif isinstance(statement, BackendFor):
                    write_ops += recurse_eval_write_ops(statement._body,
                                                        expr_counter,
                                                        var_names + [statement._var],
                                                        var_limits + [statement._limits])
                else:
                    raise NotImplementedError
                expr_counter += statement.num_exprs
            return write_ops
        return recurse_eval_write_ops(self._body, 0, [], [])

    def __str__(self):
        str_repr = "def {0}():\n".format(self.name)
        for stmt in self._body:
            new_stmt_str = ["\t" + x for x in str(stmt).split("\n")]
            for substmt in new_stmt_str:
                str_repr += substmt + "\n"
        return str_repr






class BackendGenerator(ast.NodeVisitor):
    # Public interface. Leave this alone.
    @staticmethod
    def generate_program(func):
        visitor = BackendGenerator()
        return visitor.visit(func)
        # Public interface. Leave this alone.

    def __init__(self):
        super().__init__()
        self.symbols = {}
        self.count = 0

    def visit_FuncDef(self, node):
        [self.visit(x) for x in node.body]
        body = [self.visit(x) for x in node.body]
        return_expr = OperatorExpr(self.count, RemoteReturn, [], [], [], [])
        body.append(return_expr)
        self.program = Program(node.name, body, symbols=self.symbols)
        self.program.return_expr = return_expr
        return self.program

    def visit_If(self, node):
        cond = self.visit(node.cond)
        ifbody = [self.visit(x) for x in node.body]
        elsebody = [self.visit(x) for x in node.elsebody]
        return BackendIf(cond, ifbody, elsebody)


    def visit_Lambdapack(self, node):
        compute_expr = COMPUTE_EXPR_MAP.get(node.compute)
        if (compute_expr is None):
            raise LambdaPackBackendGenerationException("Unknown operation : {0}".format(node.compute))

        reads = [self.visit(x) for x in node.reads]
        #TODO support multi-writes?
        write = self.visit(node.writes[0])
        kwargs = {k:self.visit(v) for k,v in node.options.items()}
        symbols_seen = tuple(self.symbols.values())
        matrix, idxs = write

        opexpr = OperatorExpr(self.count, compute_expr, reads, write, node.is_input, node.is_output, **kwargs)
        self.count += 1
        return opexpr

    def visit_BigMatrixBlock(self, node):
        bigm = node.bigm
        sympy_idx_expr = [self.visit(x) for x in node.bidx]
        return (bigm, sympy_idx_expr)

    def visit_Ref(self, node):
        s = sympy.Symbol(node.name)
        self.symbols[str(s)] = s
        return s

    def visit_UnOp(self, node):
        val = self.visit(node.e)
        if (node.op == "Neg"):
            return sympy.Mul(-1, val)

    def visit_IntConst(self, node):
        return sympy.Integer(node.val)

    def visit_FloatConst(self, node):
        return sympy.Float(node.val)

    def visit_Reduce(self, node):
        var = self.visit(node.var)
        expr = self.visit(node.expr)
        start = self.visit(node.start)
        end  = self.visit(node.end)
        raise Exception("Not implemented yet...")
        return ReductionTree(var=var, expr=expr, limits=[start,end], op=node.op)


    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if node.op == "Add":
            return sympy.Add(lhs,rhs)
        elif node.op =="Sub":
            return sympy.Add(lhs,-1*rhs)
        elif node.op =="Mul":
            return sympy.Mul(lhs,rhs)
        elif node.op == "Div":
            return sympy.Mul(lhs,sympy.Pow(rhs, -1))

    def visit_For(self, node):
        loop_var = sympy.Symbol(node.var)
        min_idx = self.visit(node.min)
        max_idx = self.visit(node.max)
        all_symbols = tuple(self.symbols.values())
        body = [self.visit(x) for x in node.body]
        return BackendFor(var=loop_var, limits=[min_idx, max_idx], body=body)

def cholesky_with_if(I:BigMatrix, O:BigMatrix, N:int):
    for i in range(n):
        O[-1,i,i] = cholesky(O[i,i,i], foo=True)

    if (i == 0):
        for j in range(i,n):
            O[-1,i,j] = trsm(I[-1,i,i], I[i,i,i+j])
        for j in range(i,n):
            for k in range(j,n):
                O[i+1,j,k] = syrk(I[i,j,k], I[i,i,j], I[i,i,k])
    else:
        for j in range(i,n):
            O[-1,i,j] = trsm(O[-1,i,i], O[i,i,i+j])
        for j in range(i,n):
            for k in range(j,n):
                O[i+1,j,k] = syrk(O[i,j,k], O[i,i,j], O[i,i,k])



class BackendIf(BlockStatement):
    def __init__(self, cond_expr, ifbody, elsebody):
        self._var = var
        self._limits = limits
        self._ifbody = ifbody
        self._elsebody = elsebody
        super().__init__([ifbody, elsebody])

    def __str__(self):
        str_repr = "if {0}:".format(self.cond_expr)
        str_repr += "\n"
        for stmt in self._ifbody:
            str_repr += "\n".join(["\t" + x for x in str(stmt).split("\n")])
        if (len(self._elsebody) > 0):
            str_repr += "else:"
        for stmt in self._elsebody:
            str_repr += "\n".join(["\t" + x for x in str(stmt).split("\n")])
        return str_repr

class BackendFor(BlockStatement):
    def __init__(self, var, limits, body):
        self._var = var
        self._limits = limits
        super().__init__(body)

    def __str__(self):
        str_repr = "for {0} in range({1},{2}):".format(self._var, self._limits[0], self._limits[1])
        str_repr += "\n"
        for stmt in self._body:
            new_stmt_str = ["\t" + x for x in str(stmt).split("\n")]
            for substmt in new_stmt_str:
                str_repr += substmt + "\n"
        return str_repr

class ReductionTree(BlockStatement):
    def __init__(self, var, limits, op, expr):
        self._var = var
        self._limits = limits
        self._op = op
        self._expr = expr
        super().__init__([])

    def __str__(self):
        str_repr = "reduce(expr={0}{1}, var={2}, start={3}, end={4}, op={5})".format(self._expr[0].key, self._expr[1], self._var, self._limits[0], self._limits[1], self._op)
        return str_repr

def make_3d_input_parent_fn(I):
    async def parent_fn_async(self, loop, *bidxs):
        if bidxs[0] == 0:
            #print("CALLING PARENT FN")
            return (await I.get_block_async(loop, *bidxs[1:]))[np.newaxis]
        else:
            exist = self.block_idxs_exist
            raise Exception("This shouldn't happen {0}, {1}".format(bidxs, exist))
            return constant_zeros(self, *bidxs)[np.newaxis]
    return parent_fn_async

def forward_sub(x:BigMatrix, L:BigMatrix, b:BigMatrix, S:BigMatrix, N:int):
    for i in range(N):
        S[i,0] = b[i]
        for j in range(1, i):
            S[i,j] = syrk(S[i,j-1], L[i,j], x[j])
        x[i] = trsm(S[i,i], L[i,i])

def forward_sub(x:BigMatrix, L:BigMatrix, b:BigMatrix, S:BigMatrix, N:int):
    for i in range(N):
        S[i,0] = b[i]
        for j in range(1, i):
            S[i,j] = gemm(L[i,j], x[j])
        S[i,N] = reduce(expr=S[i,j], var=j, start=1, end=i, op="sum")
        x[i] = trsm(S[i,N], L[i,i])

def backward_sub(x:BigMatrix, L:BigMatrix, b:BigMatrix, S:BigMatrix, N:int):
    for i in range(N):
        S[i,0] = b[N - 1- i]
        for j in range(1, i):
            S[i,j] = syrk(S[i,j-1], L[N - 1 - i, N - 1 - j], x[N - 1 - j])
        x[i] = trsm(S[i,i], L[i,i])


def cholesky(O:BigMatrix, I:BigMatrix, S:BigMatrix,  N:int, truncate:int):
    # handle first loop differently
    O[0,0] = cholesky(I[0,0])
    for j in range(1,N - truncate):
        O[j,0] = trsm(O[0,0], I[j,0])
        for k in range(1,j+1):
            S[1,j,k] = syrk(I[j,k], O[j,0], O[k,0])

    for i in range(1,N - truncate):
        O[i,i] = cholesky(S[i,i,i])
        for j in range(i+1,N - truncate):
            O[j,i] = trsm(O[i,i], S[i,j,i])
            for k in range(i+1,j+1):
                S[i+1,j,k] = syrk(S[i,j,k], O[j,i], O[k,i])


def _forward_solve(L,b,out_bucket=None, truncate=0):
    S = BigMatrix("Fsolve.Intermediate({0})".format(L.key), shape=(L.shape[0], L.shape[0]), shard_sizes=(L.shard_sizes[0], L.shard_sizes[0]), write_header=True)
    X = BigMatrix("FSolve({0}, {1})".format(L.key, b.key), shape=(L.shape[0],1), shard_sizes=(L.shard_sizes[0],1), write_header=True)
    program = lpcompile(forward_sub, inputs=["L", "b"], outputs=["x"])(x=X,S=S,b=b,L=L, N=int(np.ceil(X.shape[0]/X.shard_sizes[0])))
    print(program.starters)
    print(program.unroll_program())

def _chol(X, out_bucket=None, truncate=0):
    S = BigMatrix("Cholesky.Intermediate({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
    O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.shape[0], X.shape[0]), shard_sizes=(X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
    O.parent_fn = dill.dumps(constant_zeros)
    block_len = len(X._block_idxs(0))
    program = lpcompile(cholesky, inputs=["I"], outputs=["O"])(O=O,I=X,S=S,N=int(np.ceil(X.shape[0]/X.shard_sizes[0])), truncate=truncate)
    logging.debug("Starters: " + str(program.find_starters()))
    starters = program.find_starters()
    logging.debug("Terminators: " + str(program.find_terminators()))
    operator_expr = program.get_expr(starters[0][0])
    inst_block = operator_expr.eval_operator(starters[0][1])

    #print("program size", len(program.unroll_program()))
    instrs = inst_block.instrs
    program.starters = starters
    return program, S, O

if __name__ == "__main__":
    N = 65536*32
    print("Problem size", N)
    I = BigMatrix("CholeskyInput", shape=(int(N),int(N)), shard_sizes=(4096, 4096), write_header=True)
    program, S, O = _chol(I)
    print(program)
    s = time.time()
    c = program.get_children(0, {})
    e = time.time()
    print("1st cholesky child time: {0}".format(e - s))
    print("1st cholesky num children: {0}".format(len(c)))
    #print("1st cholesky children: {0}".format(c))

    s = time.time()
    c = program.get_children(3, {"i":1})
    e = time.time()
    print("2nd cholesky child time: {0}".format(e - s))
    print("2nd cholesky num children: {0}".format(len(c)))
    #print("2nd cholesky children: {0}".format(c))

    #L = BigMatrix("SolveInput", shape=(int(N),int(N)), shard_sizes=(4096, 4096), write_header=True)
    #b = BigMatrix("SolveRHS", shape=(int(N),), shard_sizes=(4096,), write_header=True)
    #_forward_solve(L, b)
