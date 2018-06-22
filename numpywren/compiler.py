import ast
import abc
import inspect
import astor
import sympy
from numpywren.matrix import BigMatrix
from numpywren import exceptions, utils
from numpywren.lambdapack import RemoteCholesky, RemoteTRSM, RemoteSYRK, RemoteRead, RemoteWrite, InstructionBlock, RemoteReturn
from numpywren.matrix_utils import constant_zeros
import numpy as np

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

VALID_CALLS = ['cholesky','syrk','trsm']
COMPUTE_EXPR_MAP = {"cholesky": RemoteCholesky, "syrk": RemoteSYRK, "trsm":RemoteTRSM}

VALID_TYPES = {'BigMatrix', 'int'}

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
    _fields = ['var', 'min', 'max', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']

class NumpywrenRead(ast.AST):
    _fields = ['matrix_block']

class NumpywrenWrite(ast.AST):
    _fields = ['matrix_block']

class BigMatrixBlock(ast.AST):
    _fields = ['bigm', 'bidx']

class Lambdapack(ast.AST):
    _fields = ['compute', 'reads', 'writes', 'options']



class NumpywrenParse(ast.NodeVisitor):
    """
    Translate a lambdapack expression.
    """
    def __init__(self, args):
        super().__init__()
        self.input_args = args
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

        assert(isinstance(node.value, ast.Call))
        assert(isinstance(node.targets[0], ast.Subscript))
        writes = [self.visit(node.targets[0])]
        assert(node.value.func.id in VALID_CALLS)
        reads = []
        for arg in node.value.args:
            assert(isinstance(arg, ast.Subscript))
            reads.append(self.visit(arg))
        options = {}
        for kw in node.value.keywords:
            options[kw.arg] = self.visit(kw.value)

        return Lambdapack(node.value.func.id, reads, writes, options)

    def visit_If(self, node):
        cond = self.visit(node.test)
        body = [self.visit(x) for x in node.body]
        else_body = [self.visit(x) for x in node.orelse]
        return If(cond, body, else_body)

    def visit_FunctionDef(self, func):
        args = [x.arg for x in func.args.args]
        annotations = [x.annotation for x in func.args.args]
        if (None in annotations):
            raise exceptions.LambdaPackParsingException("All function arguments must typed")
        annotations = [x.id for x in annotations]

        if (set(annotations) != VALID_TYPES):
            raise exceptions.LambdaPackParsingException("All function arguments must be of types: {0}".format(VALID_TYPES))

        for arg_name, arg in zip(args,self.input_args):
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
        else:
            is_range = False

        if (not is_range):
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

        return For(var, start, end, body)

    def visit_Subscript(self, node):
        assert(isinstance(node.slice.value, ast.Tuple))
        if (node.value.id not in self.symbols):
            raise exceptions.LambdaPackParsingException("Unknown BigMatrix references")

        return BigMatrixBlock(self.symbols[node.value.id], [self.visit(x) for x in node.slice.value.elts])





def lpcompile(function):
    def func(*args, **kwargs):
        print(function)
        print(inspect.getsource(function))
        function_ast = ast.parse(inspect.getsource(function)).body[0]
        print("Python AST:\n{}\n".format(astor.dump(function_ast)))
        parser = NumpywrenParse(args)
        lp_ast = parser.visit(function_ast)
        print("Source code: \n{}\n".format(inspect.getsource(function)))
        print("IR:\n{}\n".format(astor.dump_tree(lp_ast)))
        program = BackendGenerator.generate_program(lp_ast)
        print(program)
        return program
    return func


class Statement(abc.ABC):
    @abc.abstractmethod
    def get_expr(self, index):
        pass


class OperatorExpr(Statement):
    def __init__(self, opid, compute_expr, read_exprs, write_expr, **options):
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

    def get_expr(self, index):
        assert index == 0
        return self

    def eval_operator(self, var_values):
        read_instrs = self.eval_read_instrs(var_values)
        compute_instr = self._eval_compute_instr(read_instrs)
        print("var values", var_values)
        print("compute instr", compute_instr)
        write_instr = self.eval_write_instr(compute_instr, var_values)
        return InstructionBlock(read_instrs + [compute_instr, write_instr], priority=0)

    def _eval_compute_instr(self, read_instrs):
        return self.compute_expr(self.opid, read_instrs, **self.options)

    def find_reader_var_values(self, write_ref, var_names, var_limits):
        results = []
        for read_expr in self._read_exprs:
            if write_ref[0] != read_expr[0]:
                continue
            if not var_names:
                if write_ref[1] == read_expr[1]:
                    results.append({})
                continue
            assert len(write_ref[1]) == len(read_expr[1])
            results += self._enumerate_possibilities(write_ref[1], read_expr[1],
                                                     var_names, var_limits)
        return utils.remove_duplicates(results)

    def find_writer_var_values(self, read_ref, var_names, var_limits):
        if read_ref[0] != self._write_expr[0]:
            return []
        if not var_names:
            if read_ref[1] == self._write_expr[1]:
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
        def recurse_enum_pos(idx, var_limits, parametric_eqns):
            if not parametric_eqns:
                return []
            if len(parametric_eqns) == len(var_names):
                var_values = {}
                value = []
                #print("limits",var_limits)
                new_limits = [(lim[0].subs(parametric_eqns), lim[1].subs(parametric_eqns))
                               for lim in var_limits]
                for var, limit in zip(var_names, new_limits):
                    value = parametric_eqns[var]
                    if not value.is_integer: 
                        return []
                    assert parametric_eqns[var].is_integer
                    assert limit[0].is_integer
                    assert limit[1].is_integer
                    if parametric_eqns[var] < limit[0] or parametric_eqns[var] >= limit[1]:
                        return []
                    var_values[var] = int(value)
                return [var_values]

            curr_eqn = parametric_eqns.get(var_names[idx])
            curr_limit = var_limits[idx]
            if curr_eqn is None:
                refs = []
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
        return str_expr



class BlockStatement(Statement, abc.ABC):
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

    def __str__(self):
        str_repr = ""
        for stmt in self._body:
            str_repr += str(stmt)
            str_repr += "\n"
        return str_repr




class Program(BlockStatement):
    def __init__(self, name, body=[], starters=None):
        self.name = name
        self.starters = starters
        self.symbols = {}
        super().__init__(body)

    def get_children(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        if (operator_expr == self.return_expr):
            return []
        write_ref = operator_expr.eval_write_ref(var_values)
        children = self.eval_read_operators(write_ref)
        if (len(children) == 0):
            children = [self.return_expr]
        # handle returns
        return children

    def get_parents(self, expr_idx, var_values):
        operator_expr = self.get_expr(expr_idx)
        write_ref = operator_expr.eval_write_ref(var_values)
        parents = []
        operator_expr = self.program.get_expr(expr_idx)
        read_refs = child_operator_expr.eval_read_refs(var_values)
        for read_ref in read_refs:
            parents += self.program.eval_write_operators(read_ref)
        parents = utils.remove_duplicates(parents)
        return parents

    def find_terminators(self):
        terminators = []
        def recurse_find_terminators(body, expr_counter, var_values):
            i = expr_counter
            i_start = expr_counter
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    #print("analyzing statement...", statement)
                    #print(i)
                    children = self.get_children(i, var_values)
                    #print("children", children)
                    if (len(children) == 1 and children[0] == self.return_expr):
                        terminators.append((i, var_values))
                if isinstance(statement, BackendFor):
                    start_val = statement._limits[0].subs(var_values)
                    end_val = statement._limits[1].subs(var_values)
                    var = statement._var
                    for j in range(start_val, end_val):
                        var_values_recurse =var_values.copy()
                        var_values_recurse[var] = j
                        recurse_find_terminators(statement._body, i, var_values_recurse)
                i += statement.num_exprs
            return terminators
        return utils.remove_duplicates(recurse_find_terminators(self._body, 0, {}))




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
                elif isinstance(statement, For):
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
        self.count = 0

    def visit_FuncDef(self, node):
        body = [self.visit(x) for x in node.body]
        return_expr = OperatorExpr(self.count, RemoteReturn, [], [])
        body.append(return_expr)
        self.program = Program(node.name, body=body)
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
        opexpr = OperatorExpr(self.count, compute_expr, reads, write, **kwargs)
        self.count += 1
        return opexpr

    def visit_BigMatrixBlock(self, node):
        bigm = node.bigm
        sympy_idx_expr = [self.visit(x) for x in node.bidx]
        return (bigm, sympy_idx_expr)

    def visit_Ref(self, node):
        return sympy.Symbol(node.name)

    def visit_UnOp(self, node):
        val = self.visit(node.e)
        if (node.op == "Neg"):
            return sympy.Mul(-1, val)

    def visit_IntConst(self, node):
        return sympy.Integer(node.val)

    def visit_FloatConst(self, node):
        return sympy.Float(node.val)

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


def cholesky(O:BigMatrix, N:int):
    for i in range(N):
        O[N-1,i,i] = cholesky(O[i,i,i])
        for j in range(i+1,N):
            O[N-1,i,j] = trsm(O[N-1,i,i], O[i,i,i+j])
        for j in range(i+1,N):
            for k in range(j+1,N):
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



def make_3d_input_parent_fn(I):
    def parent_fn(self, *bidxs):
        if bidxs[0] == 0:
            return I.get_block(*bidxs[1:])
        else:
            raise Exception("This shouldn't happen")
            return constant_zeros(self, *bidxs)
    return parent_fn

def make_3d_input_parent_fn_async(I):
    async def parent_fn_async(self, loop, *bidxs):
        if bidxs[0] == 0:
            print("CALLING PARENT FN")
            return (await I.get_block_async(loop, *bidxs[1:]))[np.newaxis]
        else:
            raise Exception("This shouldn't happen")
            return constant_zeros(self, *bidxs)[np.newaxis]
    return parent_fn_async



def cholesky(O:BigMatrix, N:int):
    for i in range(N):
        O[N,i,i] = cholesky(O[i,i,i])
        for j in range(i+1,N):
            O[N,i,j] = trsm(O[N,i,i], O[i,i,j])
        for j in range(i+1,N):
            for k in range(j,N):
                O[i+1,j,k] = syrk(O[i,j,k], O[N,i,j], O[N,i,k])

def _chol(X, out_bucket=None):
    O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
    block_len = len(X._block_idxs(0))
    parent_fn = make_3d_input_parent_fn(X)
    parent_fn_async = make_3d_input_parent_fn_async(X)
    O.parent_fn = parent_fn
    O.parent_fn_async = parent_fn_async
    starters = [(0, {sympy.Symbol("i"): 0})]
    program = lpcompile(cholesky)(O,int(np.ceil(X.shape[0]/X.shard_sizes[0])))
    print(program.get_children(*starters[0]))
    print(program.get_expr(3))
    print("Terminators", program.find_terminators())
    return
    operator_expr = program.get_expr(starters[0][0])
    inst_block = operator_expr.eval_operator(starters[0][1])
    instrs = inst_block.instrs
    print("matrix_parent_fn", instrs[0].matrix, instrs[0].matrix.parent_fn)
    print("matrix_parent_fn", repr(instrs[0].matrix), instrs[0].matrix.parent_fn)
    program.starters = starters
    print(repr(O), O.parent_fn)
    return program, O.submatrix(block_len), O








if __name__ == "__main__":
    N = 65536*4
    I = BigMatrix("CholeskyInput", shape=(int(N),int(N)), shard_sizes=(4096, 4096), write_header=True)
    _chol(I)
