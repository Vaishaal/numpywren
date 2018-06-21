import ast
import abc
import inspect
import astor
import sympy
from numpywren.matrix import BigMatrix
from numpywren import exceptions
from numpywren.lambdapack import RemoteCholesky, RemoteTRSM, RemoteSYRK

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

    def __init__(self, cond, body, elseBody=None):
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
    _fields = ['key', 'bidx']

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
        print(node)
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
        body = Block([self.visit(x) for x in node.body])
        else_body = Block([self.visit(x) for x in node.orelse])
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

        return BigMatrixBlock(self.symbols[node.value.id].key, [self.visit(x) for x in node.slice.value.elts])





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
        print(parser.loop_variables)
        program = BackendGenerator.generate_program(lp_ast)
        print(program)
    return func


class Statement(abc.ABC):
    @abc.abstractmethod
    def get_expr(self, index):
        pass


class OperatorExpr(Statement):
    def __init__(self, opid, compute_expr, read_exprs, write_expr, **kwargs):
        ''' compute_expr is a subclass RemoteInstruction (not an instance)
             read_exprs, and write_exprs
            are of lists of type (BigMatrix, [sympy_expr, sympy_expr...])
        '''
        self._read_exprs = read_exprs
        self._write_expr = write_expr
        self.compute_expr = compute_expr
        self.opid = opid
        self.num_exprs = 1

    def get_expr(self, index):
        assert index == 0
        return self

    def eval_operator(self, var_values):
        read_instrs = self.eval_read_instrs(var_values)
        compute_instr = self._eval_compute_instr(read_instrs) 
        write_instr = self.eval_write_instr(compute_instr, var_values)
        return InstructionBlock(read_instrs + [compute_instr, write_instr], priority=0)

    def _eval_compute_instr(self, read_instrs):
        return self.compute_expr(self.opid, read_instrs, **kwargs)

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

        write_str = ""
        for write_expr in self._write_expr:
            write_str += write_expr[0].key
            write_str += str(write_expr[1])
            write_str += ","
        write_str = write_str[:-1]

        str_expr = "{2} = {0}({1})".format(self.compute_expr.__name__, read_str, write_str)
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
    def __init__(self, body=[], starters=None):
        self.starters = starters
        self.symbols = {}
        super().__init__(body)

    def eval_read_operators(self, write_ref):
        def recurse_eval_read_ops(body, expr_counter, var_names, var_limits):
            read_ops = []
            for statement in body:
                if isinstance(statement, OperatorExpr):
                    valid_var_values = statement.find_reader_var_values(
                        write_ref, var_names, var_limits)
                    read_ops += [(expr_counter, vals) for vals in valid_var_values]
                elif isinstance(statement, For):
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
        self.program = Program(body=body)
        return self.program


    def visit_Lambdapack(self, node):
        compute_expr = COMPUTE_EXPR_MAP.get(node.compute)
        if (compute_expr is None):
            raise LambdaPackBackendGenerationException("Unknown operation : {0}".format(node.compute))

        reads = [self.visit(x) for x in node.reads]
        writes = [self.visit(x) for x in node.writes]
        kwargs = {k:self.visit(v) for k,v in node.options.items()}
        return OperatorExpr(self.count, compute_expr, reads, writes, **kwargs)
        self.count += 1

    def visit_BigMatrixBlock(self, node):
        bigm = BigMatrix(node.key)
        sympy_idx_expr = [self.visit(x) for x in node.bidx]
        return (bigm, sympy_idx_expr)

    def visit_Ref(self, node):
        return sympy.Symbol(node.name)

    def visit_UnOp(self, node):
        val = self.visit(node.e)
        if (node.op == "Neg"):
            return sympy.Mul(-1, val)

    def visit_IntConst(self, node):
        return node.val

    def visit_FloatConst(self, node):
        return node.val

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















def cholesky(O:BigMatrix, N:int):
    for i in range(n):
        O[-1,i,i] = cholesky(O[i,i,i], foo=True)
    for j in range(i,n):
        O[-1,i,j] = trsm(O[-1,i,i], O[i,i,i+j])
    for j in range(i,n):
        for k in range(j,n):
            O[i+1,j,k] = syrk(O[i,j,k], O[i,i,j], O[i,i,k])

class BackendFor(BlockStatement):
    def __init__(self, var, limits, body):
        self._var = var
        self._limits = limits
        super().__init__(body)

    def __str__(self):
        str_repr = "for {0} in range({1},{2}):".format(self._var, self._limits[0], self._limits[1])
        str_repr += "\n"
        for stmt in self._body:
            str_repr += "\n".join(["\t" + x for x in str(stmt).split("\n")])
        return str_repr













if __name__ == "__main__":
    bigm = BigMatrix("test_matrix", shape=(10,10), shard_sizes=(10,10))
    lpcompile(cholesky)(bigm, 10)
