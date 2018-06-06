import ast
import inspect
import astor

########
## IR ##
########
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
"""

FUNCS = ['chol','syrk','trsm']

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']

class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']

class UnOp(ast.AST):
    _fields = ['op', 'e']

class Ref(ast.AST):
    _fields = ['name', 'index']

    def __init__(self, name, index=None):
        return super().__init__(name, index)

class IntConst(ast.AST):
    _fields = ['val',]

class FloatConst(ast.AST):
    _fields = ['val',]

## Stmts ##
class Assign(ast.AST):
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
    _fields = ['matrix_key', 'bidx']

class Lambdapack(ast.AST):
    _fields = ['compute', 'reads', 'writes']



class NumpywrenParse(ast.NodeVisitor):
    """
    Translate a lambdapack expression.
    """
    def __init__(self):
        super().__init__()
        self.loop_variables = set()

    def visit_Name(self, node):
        return Ref(node.id)

    def visit_Assign(self, node):
        if (len(node.targets) != 1):
            raise NotImplementedError("Only single argument assignment supported")

        assert(isinstance(node.value, ast.Call))
        assert(isinstance(node.targets[0], ast.Subscript))
        writes = [self.visit(node.targets[0])]
        assert(node.value.func.id in FUNCS)
        reads = []
        for arg in node.value.args:
            assert(isinstance(arg, ast.Subscript))
            reads.append(self.visit(arg))
        return Lambdapack(node.value.func.id, reads, writes)

    def visit_FunctionDef(self, func):
        if (len(func.args.args) != 2):
            raise Exception("Only support 2 argument functions")
        args = [x.arg for x in func.args.args]
        name = func.name
        assert isinstance(func.body, list)
        body = Block([self.visit(x) for x in func.body])
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

        body = Block([self.visit(x) for x in node.body])
        var = node.target.id
        self.loop_variables.add(node.target.id)

        return For(var, start, end, body)

    def visit_Subscript(self, node):
        assert(isinstance(node.slice.value, ast.Tuple))
        return BigMatrixBlock(node.value, [x for x in node.slice.value.elts])




def lpcompile(function):
    print(function)
    print(inspect.getsource(function))
    function_ast = ast.parse(inspect.getsource(function)).body[0]
    #print("Python AST:\n{}\n".format(astor.dump(function_ast)))
    parser = NumpywrenParse()
    simpleFun = parser.visit(function_ast)
    print("Source code: \n{}\n".format(inspect.getsource(function)))
    print("IR:\n{}\n".format(astor.dump_tree(simpleFun)))
    print(parser.loop_variables)

def cholesky(O,n):
    for i in range(n):
        O[-1,i,i] = chol(O[i,i,i])
    for j in range(i,n):
        O[-1,i,j] = trsm(O[-1,i,i], O[i,i,i+j])
    for j in range(i,n):
        for k in range(j,n):
            O[i+1,j,k] = syrk(O[i,j,k], O[i,i,j], O[i,i,k])




if __name__ == "__main__":
    lpcompile(cholesky)
