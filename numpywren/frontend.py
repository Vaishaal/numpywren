
import ast
import astor
import time
import dill
import logging
import abc
from numpywren.matrix import BigMatrix
from numpywren.matrix_init import shard_matrix
from numpywren import exceptions, compiler, utils
import numpy as np
import asyncio
from numpywren.kernels import *
from collections import namedtuple
import inspect
from operator import *
from numpywren import lambdapack as lp
from scipy.linalg import qr
import sympy
from numbers import Number

op_table = {}

op_table['Add'] = add
op_table['Div'] = truediv
op_table['And'] = and_
op_table['Not'] = and_
op_table['Mult'] = mul
op_table['Mul'] = mul
op_table['Sub'] = sub
op_table['Mod'] = mod
op_table['Pow'] = pow
op_table['Or'] = or_
op_table['EQ'] = eq
op_table['NE'] = ne
op_table['Neg'] = neg
op_table['Not'] = not_
op_table['LT'] = lt
op_table['LE'] = le
op_table['GE'] = ge
op_table['GT'] = gt
op_table['ceiling'] = sympy.ceiling
op_table['floor'] = sympy.floor
op_table['log'] = sympy.log


logger = logging.getLogger('numpywren')

''' front end parser + typechecker for lambdapack

comp_op: '<'|'>'|'=='|'>='|'<='
un_op : '-' | 'not'
_op ::=  + | - | * | ** | / | // | %
var ::= NAME
term ::= NAME | INT | FLOAT | expr
un_expr ::= un_op term | term
mul_expr ::= un_expr (('/'|'*') un_expr)*
arith_expr ::= mul_expr ((‘+’|’-’) mul_expr)*
simple_expr ::= mul_expr (‘**’ mul_expr)*
comparison: expr (comp_op expr)*
mfunc_expr = mfunc(expr)
expr ::= simple_expr | m_func_expr | comparison
m_func ::=  ceiling | floor | log
index_expr ::= NAME ‘[‘ expr (, expr)* ‘]’
op := NAME
_arg := (index_expr | expr)
_index_expr_assign ::= index_expr (, index_expr)* ‘=’ op(_arg (, _arg)*
_var_assign ::= var ‘=’ expr
assign_stmt ::= _index_expr_assign | _var_assign
block := stmt (NEW_LINE stmt)*
for_stmt ::= 'for' var 'in' ‘range(‘expr, expr’)’  ':' block
if_stmt: 'if' expr ':' block  ['else' ':' block]
stmt ::= for_stmt | with_stmt | assign_stmt | expr
'''

KEYWORDS = ["ceiling", "floor", "log", "REDUCTION_LEVEL"]
M_FUNCS = ['ceiling', 'floor', 'log']
M_FUNC_OUT_TYPES = {}
M_FUNC_OUT_TYPES['ceiling'] = int
M_FUNC_OUT_TYPES['floor'] = int
M_FUNC_OUT_TYPES['log'] = float
M_FUNC_OUT_TYPES['log2'] = float

class Expression(abc.ABC):
    pass

class LambdaPackType(abc.ABC):
    pass

class LambdaPackAttributes(abc.ABC):
    pass

class NullType(LambdaPackType):
    pass

class PrimitiveType(LambdaPackType):
    pass

class NumericalType(PrimitiveType):
    pass

class IntType(NumericalType):
    pass

class LinearIntType(IntType):
    pass

class Const(LambdaPackAttributes):
    pass

class BoolType(PrimitiveType):
    pass

class ConstBoolType(BoolType, Const):
    pass

class FloatType(NumericalType):
    pass

class ConstIntType(LinearIntType, Const):
    pass

class ConstFloatType(FloatType, Const):
    pass

class IndexExprType(LambdaPackType):
    pass

class BigMatrixType(LambdaPackType):
    pass

RangeVar = namedtuple("RangeVar", ["var", "start", "end", "step"])
RemoteCallWithContext = namedtuple("RemoteCall", ["remote_call", "scope"])

## Exprs ##
class BinOp(ast.AST, Expression):
    _fields = ['op', 'left', 'right', 'type']

class CmpOp(ast.AST, Expression):
    _fields = ['op', 'left', 'right', 'type']

class UnOp(ast.AST, Expression):
    _fields = ['op', 'e', 'type']

class Mfunc(ast.AST, Expression):
    _fields = ['op', 'e', 'type']

class Assign(ast.AST):
    _fields = ['lhs', 'rhs']

class Ref(ast.AST, Expression):
    _fields = ['name', 'type']

class IntConst(ast.AST, Expression):
    _fields = ['val', 'type']

class FloatConst(ast.AST, Expression):
    _fields = ['val', 'type']

class BoolConst(ast.AST, Expression):
    _fields = ['val','type']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']

    def __init__(self, cond, body, elseBody=[]):
        return super().__init__(cond, body, elseBody)

class Attr(ast.AST):
    _fields = ['obj', 'attr_name']

class Stargs(ast.AST):
    _fields = ['args']

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'step', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body', 'arg_types']

class BigMatrixBlock(ast.AST):
    _fields = ['name', 'bigm', 'bidx', 'type']

class RemoteCall(ast.AST):
    _fields = ['compute', 'output', 'args', 'kwargs', 'type']

class Reduction(ast.AST):
    _fields = ['var', 'min', 'max', 'expr', 'b_fac', 'remote_call', 'recursion']

class IndexExpr(ast.AST):
    _fields = ['matrix_name', 'indices']

class Slice(ast.AST):
    _fields = ['low', 'high', 'step', 'type']

class Return(ast.AST):
    _fields = ['value', 'type']

class ReducerCall(ast.AST):
    _fields = ['name','function', 'args', 'type']

REDUCTION_SPECIALS = ["level", "reduce_args", "reduce_next", "reduce_idxs"]

def unify(type_list):
    if (len(type_list) == 1): return type_list[0]
    if (len(type_list) < 3):
        t0 = type_list[0]
        t1 = type_list[0]
        if (issubclass(t0, t1)):
            return t1
        elif (issubclass(t1, t0)):
            return t0
        else:
            raise exceptions.LambdaPackTypeException("Non unifiable types {0} vs {1}".format(type_list))
    else:
        t0,t1 = type_list[0], type_list[1]
        t01 = unify([t0, t1])
        return unify([t01] + type_list[2:])



class LambdaPackParse(ast.NodeVisitor):
    """
    Translate a lambdapack expression.
    """
    def __init__(self):
        self.in_if = False
        self.in_else = False
        self.for_loops = 0
        self.max_for_loop_id = -1
        self.current_for_loop = -1
        self.decl_dict = {}
        self.return_node = None
        self.in_reduction = False
        self.current_reduction_object = None
        self.reduce_next_exprs = []
        super().__init__()


    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(node.n, None)
        elif isinstance(node.n, float):
            return FloatConst(node.n, None)
        else:
            raise NotImplementedError("Only Integers and Floats supported")

    def visit_BoolOp(self, node):
        values = [self.visit(x) for x in node.values]

        left = self.visit(node.values[0])
        right = self.visit(node.values[1])
        op  = node.op
        if (isinstance(op, ast.Or)):
            op = "Or"
        elif (isinstance(op, ast.And)):
            op = "And"
        else:
            raise Exception("Invalid bool operation {0}".format(op))
        i = 1
        lhs = left
        while (i < len(values)):
            lhs = BinOp(op, lhs, values[i], None)
            i += 1
        return lhs

    def visit_BinOp(self, node):
        VALID_BOPS  = ["Add", "Sub", "Mult", "Div", "Mod",  "Pow", "FloorDiv", "And", "Or"]
        left = self.visit(node.left)
        right = self.visit(node.right)
        op  = node.op.__class__.__name__
        if (op not in VALID_BOPS):
            raise NotImplementedError("Unsupported BinOp {0}".format(op))
        ret = BinOp(op, left, right, None)
        return ret

    def visit_Str(self, node):
        raise NotImplementedError("Stings not supported")

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
        return CmpOp(s_op, left, right, None)

    def visit_UnaryOp(self, node):
        e = self.visit(node.operand)

        op = node.op.__class__.__name__
        if (op == "USub"):
            s_op = "Neg"
        elif (op == "Not"):
            s_op = "Not"
        else:
            raise NotImplementedError("Unsupported unary operation {0}".format(op))
        return UnOp(s_op, e, None)


    def visit_Name(self, node):
        return Ref(node.id, None)

    def visit_NameConstant(self, node):
        if (node.value == True):
            return IntConst(1, None)
        elif (node.value == False):
            return IntConst(0, None)
        else:
            raise exceptions.LambdaPackParsingException("Unsupported Name constant")


    def visit_Attribute(self, node):
        assert self.in_reduction, "Only Valid Attribute calls are to reducers"
        name = node.value.id
        assert self.decl_dict[name] == self.current_reduction_object, "Incorrect use of reduction features"
        assert node.attr in REDUCTION_SPECIALS, "Only a few special reduction special function calls are valid : {0}".format(REDUCTION_SPECIALS)
        return ReducerCall(name, node.attr, None, None)

    def visit_Call(self, node):
        func = self.visit(node.func)
        kwargs = {x.arg : self.visit(x.value) for x in node.keywords}
        args = [self.visit(x) for x in node.args]
        if (isinstance(func, ReducerCall)):
            return ReducerCall(func.name, func.function, args, None)

        if (isinstance(func, Ref)):
            if (func.name in M_FUNCS):
                assert len(node.args) == 1, "m_func calls must single argument"
                return Mfunc(func.name, self.visit(node.args[0]), None)
            else:
                try:
                    #TODO do this without eval
                    node_func_obj = eval(func.name)
                    if (callable(node_func_obj)):
                        args = [self.visit(x) for x in node.args]
                        return RemoteCall(node_func_obj, None, args, None, None)
                except NameError:
                    pass
        raise Exception("unsupported function {0}".format(func.name))

    def visit_Assign(self, node):
        rhs = self.visit(node.value)
        if (isinstance(rhs, Expression)):
            if (len(node.targets) != 1):
                raise NotImplementedError("Multiple targets only supported for RemoteOps")
            lhs = self.visit(node.targets[0])
            assign = Assign(lhs, rhs)
            if (self.in_if):
                self.decl_dict[lhs.name]  = rhs
            elif (self.in_else):
                if (lhs.name not in self.decl_dict):
                    raise exceptions.LambdaPackParsingException("Variable {0} declared in else but not in if".format(lhs.name))
                del self.decl_dict[lhs.name]
            else:
                if (lhs.name) in self.decl_dict:
                    raise exceptions.LambdaPackParsingException("multiple variable declarations forbidden")
                self.decl_dict[lhs.name] = rhs
            return assign
        elif isinstance(rhs, RemoteCall):
            lhs = self.visit(node.targets[0])
            return RemoteCall(rhs.compute, lhs, rhs.args, rhs.kwargs, rhs.type)
        else:
            raise NotImplementedError("Only assignments of expressions and remote calls supported")


    def visit_If(self, node):
        cond = self.visit(node.test)
        self.in_if = True
        tmp_decl_dict = self.decl_dict.copy()
        body = [self.visit(x) for x in node.body]
        tmp_decl_dict_2 = self.decl_dict.copy()
        self.in_if = False
        self.in_else = True
        else_body = [self.visit(x) for x in node.orelse]
        self.in_else = False
        tmp_decl_dict_3 = self.decl_dict.copy()
        if (list(tmp_decl_dict_3.keys())  != list(tmp_decl_dict.keys())):
            raise exceptions.LambdaPackParsingException("if/else didn't have symmetric pair of declarations")
        for k,v in tmp_decl_dict_2.items():
            if (k in tmp_decl_dict) and (tmp_decl_dict[k] is not tmp_decl_dict_2[k]):
                raise exceptions.LambdaPackParsingException("repeat decl in if clause: {0}".format(k))
            self.decl_dict[k] = v
        return If(cond, body, else_body)

    def visit_FunctionDef(self, func):
        args = [x.arg for x in func.args.args]
        if (len(set(args)) != len(args)):
            raise exceptions.LambdaPackParsingException("No repeat arguments allowed")
        annotations = [eval(x.annotation.id) for x in func.args.args]
        name = func.name
        assert isinstance(func.body, list)
        body = [self.visit(x) for x in func.body]
        #assert func.returns is not None, "LambdaPack functions must have explicit return type"
        return FuncDef(func.name, args, body, annotations)

    def visit_Starred(self, node):
        return Stargs(self.visit(node.value))

    def visit_For(self, node):
        iter_node = node.iter
        prev_for = self.current_for_loop
        self.for_loops += 1
        self.current_for_loop += 1
        is_call = isinstance(iter_node, ast.Call)
        if (is_call):
            is_range = iter_node.func.id == "range"
        else:
            is_range = False

        if (not is_range):
            raise NotImplementedError("Only for(x in range(...)) loops allowed")


        if (len(iter_node.args) == 1):
            start = IntConst(0, None)
            end = self.visit(iter_node.args[0])
        else:
            start = self.visit(iter_node.args[0])
            end = self.visit(iter_node.args[1])
        if (len(iter_node.args) < 3):
            step = IntConst(1, None)
        else:
            step = self.visit(iter_node.args[2])

        body = [self.visit(x) for x in node.body]
        var = node.target.id
        self.decl_dict[var] = iter_node
        self.current_for_loop = prev_for
        self.for_loops -= 1
        self.current_for_loop -= 1
        return For(var, start, end, step, body)

    def visit_Return(self, node):
        raise exceptions.LambdaPackParsingException("returns forbidden in lambdapack, pass in outputs as function arguments")


    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Tuple(self, node):
        return [self.visit(x) for x in node.elts]

    def visit_ExtSlice(self, node):
        return [self.visit(x) for x in node.dims]

    def visit_Subscript(self, node):
        index = node.slice
        matrix_id = node.value.id
        idxs = self.visit(index)
        return IndexExpr(matrix_id, idxs)

    def visit_Slice(self, node):
        if (node.lower is not None):
            low = self.visit(node.lower)
        else:
            low = None
        if (node.upper is not None):
            high = self.visit(node.upper)
        else:
            high = None
        if (node.step is not None):
            step = self.visit(node.step)
        else:
            step = None
        return Slice(low, high, step, None)

    def visit_Expr(self, node):
        return self.visit(node.value)


def python_type_to_lp_type(p_type, const=False):
    if (p_type is None):
        return NullType
    if (issubclass(p_type, int)):
        if (const):
            return ConstIntType
        else:
            return IntType
    elif (issubclass(p_type, float)):
        if (const):
            return ConstFloatType
        else:
            return FloatType
    elif (issubclass(p_type, bool)):
        return BoolType
    elif (issubclass(p_type, BigMatrix)):
        return BigMatrixType
    else:
        raise exceptions.LambdaPackTypeException("Unsupported Python type: {0}".format(p_type))



class LambdaPackTypeCheck(ast.NodeVisitor):
    ''' Assign a type to every node or throw TypeError
        * For loop bounds needs to be an integer
        * Reduction bounds need to be integers
        * Input to IndexExprs must be a LinearIntType
        * LinearIntType (*|/) ConstIntType -> LinearIntType
        * LinearIntType (//|%|**) ConstIntType -> IntType
        * LinearIntType (*|/|//|%) LinearIntType -> IntType
        * LinearIntType (+/-) LinearIntType -> LinearIntType
        * MFunc(ConstIntType) -> (ConstFloatType, ConstIntType)
        * MFunc(LinearIntType) -> (IntType, FloatType)
    '''
    def __init__(self):
        self.decl_types = {}
        pass

    def visit_FuncDef(self, func):
        annotations = [python_type_to_lp_type(x, const=True) for x in func.arg_types]

        args = [x for x in func.args]
        for arg,anot in zip(args, annotations):
            self.decl_types[arg] = anot
        body = [self.visit(x) for x in func.body]
        return FuncDef(func.name, args, body, annotations)

    def visit_Ref(self, node):
        decl_type = self.decl_types[node.name]
        if (decl_type is None):
            raise LambdaPackTypeException("Refs must be typed")
        return Ref(node.name, decl_type)


    def visit_Assign(self, node):
        rhs = self.visit(node.rhs)
        lhs = node.lhs
        if (lhs.name in self.decl_types):
            is_subclass = issubclass(rhs.type, self.decl_types[lhs.name])
            is_superclass  = issubclass(self.decl_types[lhs.name], rhs.type)
            if ((not is_subclass) and (not is_superclass)):
                raise exceptions.LambdaPackTypeException("Variables must be of unifiable type, {0} is type {1} but was assigned {2}".format(lhs.name, self.decl_types[lhs.name], rhs.type))
        else:
            self.decl_types[lhs.name] = rhs.type
        lhs = self.visit(node.lhs)
        return Assign(lhs, rhs)

    def visit_If(self, node):
        cond = self.visit(node.cond)
        if (not issubclass(cond.type, BoolType)):
            raise exceptions.LambdaPackTypeException("cond of if statement must be BoolType")
        body = [self.visit(x) for x in node.body]
        else_body = [self.visit(x) for x in node.elseBody]
        return If(cond, body, else_body)

    def visit_BinOp(self, node):
        right = self.visit(node.right)
        left = self.visit(node.left)
        r_type = right.type
        l_type = left.type
        op = node.op
        if (op == "Or" or op == "And"):
            assert(issubclass(left.type, BoolType))
            assert(issubclass(right.type, BoolType))
            out_type = BoolType
        else:
            if ((r_type is None) or (l_type is None)):
                raise LambdaPackTypeException("BinOp arguments must be typed")
            type_set = set([r_type, l_type])
            for t in type_set:
                if (not issubclass(t, NumericalType)):
                    raise LambdaPackTypeException("BinOp arguments must be Numerical")
            if (op == "Add" or op == "Sub"):
                # arith type algebra
                if (issubclass(r_type, ConstIntType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstIntType
                elif (issubclass(r_type, LinearIntType) and issubclass(l_type, LinearIntType)):
                    out_type = LinearIntType
                elif (issubclass(r_type, IntType) and issubclass(l_type, IntType)):
                    out_type = IntType
                elif (issubclass(r_type, ConstFloatType) and issubclass(l_type, ConstFloatType)):
                    out_type = ConstFloatType
                elif (issubclass(r_type, ConstFloatType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstFloatType
                elif (issubclass(l_type, ConstFloatType) and issubclass(r_type, ConstIntType)):
                    out_type = ConstFloatType
                elif (issubclass(r_type, FloatType) or issubclass(l_type, FloatType)):
                    out_type = FloatType
                else:
                    raise exceptions.LambdaPackTypeException("Unsupported type combination for add/sub")
            elif (op =="Mult"):
                # mul type algebra
                if (issubclass(r_type, LinearIntType) and issubclass(l_type, ConstIntType)):
                    out_type = LinearIntType
                if (issubclass(r_type, LinearIntType) and issubclass(l_type, LinearIntType)):
                    out_type = IntType
                elif (issubclass(r_type, IntType) and issubclass(l_type, IntType)):
                    out_type = IntType
                elif (issubclass(r_type, ConstFloatType) and issubclass(l_type, ConstFloatType)):
                    out_type = ConstFloatType
                elif (issubclass(r_type, ConstFloatType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstFloatType
                elif (issubclass(r_type, FloatType) or issubclass(l_type, FloatType)):
                    out_type = FloatType
                else:
                    raise exceptions.LambdaPackTypeException("Unsupported type combination for mul")
            elif (op =="Div"):
                # div type algebra
                if (issubclass(r_type, LinearIntType) and issubclass(l_type, ConstIntType)):
                    out_type = LinearIntType
                elif (issubclass(r_type, Const) and issubclass(l_type, Const)):
                    out_type = ConstFloatType
                else:
                    out_type = FloatType
            elif (op == "Mod"):
                if (issubclass(r_type, ConstIntType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstIntType
                elif (issubclass(r_type, IntType) and issubclass(l_type, IntType)):
                    out_type = IntType
                else:
                    out_type = FloatType
            elif (op == "Pow"):
                if (issubclass(r_type, ConstIntType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstIntType
                elif (issubclass(r_type, IntType) and issubclass(l_type, IntType)):
                    out_type = IntType
                elif (issubclass(r_type, Const) and issubclass(l_type, Const)):
                    out_type = ConstFloatType
                else:
                    out_type = FloatType
            elif (op == "FloorDiv"):
                if (issubclass(r_type, ConstIntType) and issubclass(l_type, ConstIntType)):
                    out_type = ConstIntType
                else:
                    out_type = IntType
        return BinOp(node.op, left, right, out_type)

    def visit_Return(self, node):
        r_vals = []
        if (isinstance(node.value, list)):
            for v in node.value:
                r_vals.append(self.visit(v))
        else:
            r_vals.append(self.visit(node.value))
        self.return_node_type = unify([x.type for x in r_vals])
        return Return(r_vals, self.return_node_type)

    def visit_IndexExpr(self, node):
        if (isinstance(node.indices, list)):
            idxs = [self.visit(x) for x in node.indices]
        else:
            idxs = [self.visit(node.indices)]

        out_type = unify([x.type for x in idxs])
        if (not issubclass(out_type, IntType)):
            raise exceptions.LambdaPackTypeException("Indices in IndexExprs must all of type LinearIntType {0}[{1}]".format(node.matrix_name, [str(x) for x in node.indices]))
        return IndexExpr(node.matrix_name, idxs)

    def visit_Stargs(self, node):
       args = self.visit(node.args)
       return Stargs(args)

    def visit_Slice(self, node):
        if (node.low is not None):
            low = self.visit(node.low)
            low_type = low.type
        else:
            low = None
            low_type = LinearIntType
        if (node.high is not None):
            high = self.visit(node.high)
            high_type = high.type
        else:
            high = None
            high_type = LinearIntType
        if (node.step is not None):
            step = self.visit(node.step)
            step_type = step.type
        else:
            step = None
            step_type = LinearIntType
        out_type = unify([low_type, high_type, step_type])
        return Slice(low, high, step, out_type)

    def visit_RemoteCall(self, node):
        args = [self.visit(x) for x in node.args]
        if (isinstance(node.output, list)):
            outs = [self.visit(x) for x in node.output]
        else:
            outs = [self.visit(node.output)]

        if (node.kwargs is not None):
            kwargs = {k: self.visit(v) for (k,v) in node.kwargs.items()}
        else:
            kwargs = None

        return RemoteCall(node.compute, outs, args, kwargs, type)

    def visit_Mfunc(self, node):
        vals = self.visit(node.e)
        func_type = M_FUNC_OUT_TYPES[node.op]
        if (issubclass(vals.type, Const)):
            out_type = python_type_to_lp_type(func_type, const=True)
        else:
            out_type = python_type_to_lp_type(func_type)
        return Mfunc(node.op, vals, out_type)

    def visit_CmpOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if (issubclass(lhs.type, Const) and issubclass(rhs.type, Const)):
            out_type = ConstBoolType
        else:
            out_type = BoolType
        return CmpOp(node.op, lhs, rhs, out_type)

    def visit_IntConst(self, node):
        return IntConst(node.val, ConstIntType)

    def visit_FloatConst(self, node):
        return FloatConst(node.val, ConstFloatType)

    def visit_For(self, node):
        self.decl_types[node.var] = LinearIntType
        min_idx = self.visit(node.min)
        max_idx = self.visit(node.max)
        linear_max = issubclass(min_idx.type, LinearIntType)
        linear_min = issubclass(min_idx.type, LinearIntType)
        if ((not linear_max) or (not linear_min)):
            raise LambdaPackTypeExceptions("Loop bounds must be LinearIntType")

        step = self.visit(node.step)
        body = [self.visit(x) for x in node.body]
        return For(node.var, min_idx, max_idx, step, body)

    def visit_Reduction(self, node):
        self.decl_types[node.var] = LinearIntType
        min_idx = self.visit(node.min)
        max_idx = self.visit(node.max)
        if (isinstance(node.expr, list)):
            expr = [self.visit(x) for x in node.expr]
        else:
            expr = [self.visit(node.expr)]
        for expr_i in expr:
            if (not isinstance(expr_i, IndexExpr)):
                raise LambdaPackTypeExceptions("Reduction Exprs must be of IndexExpr type")
        b_fac = self.visit(node.b_fac)
        remote_call = self.visit(node.remote_call)
        recursion = [self.visit(x) for x in node.recursion]
        return Reduction(node.var, min_idx, max_idx, expr, b_fac, remote_call, recursion)

class BackendGenerate(ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.global_scope = {}
        self.current_scope = self.global_scope
        self.all_symbols = {}
        self.remote_calls = {}
        self.evaluators = {}
        self.max_calls = 0
        self.count = 0
        self.arg_values = args
        self.kwargs = kwargs
        self.for_depth = 0

    def visit_FuncDef(self, node):
        assert len(node.args) == len(self.arg_values), "function {0} expected {1} args got {2}".format(node.name, len(node.args), len(self.arg_values))
        print("ARG VALUES", self.arg_values)
        for i, (arg, arg_value, arg_type)  in enumerate(zip(node.args, self.arg_values, node.arg_types)):
            p_type = python_type_to_lp_type(type(arg_value), const=True)
            if (not issubclass(p_type, arg_type)):
                raise LambdaPackBackendGenerationException("arg {0} wrong type expected {1} got {2}".format(i, arg_type, p_type))
            self.global_scope[arg] = arg_value
            self.all_symbols[arg] = arg_value
        body = [self.visit(x) for x in node.body]
        assert(len(node.args) == len(self.arg_values))

    def visit_RemoteCall(self, node):
        reads = [self.visit(x) for x in node.args]
        writes = [self.visit(x) for x in node.output]
        self.remote_calls[self.max_calls] = RemoteCallWithContext(node, self.current_scope)
        self.max_calls += 1
        return node

    def visit_If(self, node):
        cond = self.visit(node.cond)
        prev_scope = self.current_scope
        self.current_conds.append(cond)
        if_scope = {"__parent__": prev_scope, "__condtrue__": cond}
        self.current_scope = if_scope
        if_body = [self.visit(x) for x in node.body]
        else_scope = {"__parent__": prev_scope, "__condfalse__": cond}
        self.current_scope = else_scope
        else_body = [self.visit(x) for x in node.elseBody]
        self.current_scope = prev_scope
        return node

    def visit_CmpOp(self, node):
        return node

    def visit_IndexExpr(self, node):
        return node

    def visit_Mfunc(self, node):
        return node

    def visit_BinOp(self, node):
        return node

    def visit_Assign(self, node):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        self.all_symbols[lhs] = rhs
        self.current_scope[str(lhs)] = rhs
        return node

    def visit_For(self, node):
        prev_scope = self.current_scope
        self.for_depth += 1
        for_loop_scope = {"__parent__": prev_scope, 'depth': self.for_depth}

        self.current_scope = for_loop_scope
        min_idx = self.visit(node.min)
        max_idx = self.visit(node.max)
        step = self.visit(node.step)
        for_loop_scope[node.var] = RangeVar(node.var, min_idx, max_idx, step)
        body = [self.visit(x) for x in node.body]
        self.for_depth -= 1
        self.current_scope = prev_scope
        return node

    def visit_Ref(self, node):
        return node.name

    def visit_UnOp(self, node):
        return node

    def visit_IntConst(self, node):
        return node

    def visit_FloatConst(self, node):
        return node


def lpcompile(function):
    function_ast = ast.parse(inspect.getsource(function)).body[0]
    logging.debug("Python AST:\n{}\n".format(astor.dump(function_ast)))
    parser = LambdaPackParse()
    type_checker = LambdaPackTypeCheck()
    lp_ast = parser.visit(function_ast)
    logging.debug("IR AST:\n{}\n".format(astor.dump_tree(lp_ast)))
    lp_ast_type_checked = type_checker.visit(lp_ast)
    logging.debug("typed IR AST:\n{}\n".format(astor.dump_tree(lp_ast_type_checked)))
    print("typed IR AST:\n{}\n".format(astor.dump_tree(lp_ast_type_checked)))
    def f(*args, **kwargs):
        backend_generator = BackendGenerate(*args, **kwargs)
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

def eval_index_expr(index_expr, scope):
    bigm = scope_lookup(index_expr.matrix_name, scope)
    idxs = []
    for index in index_expr.indices:
        idxs.append(eval_expr(index, scope))
    return bigm, tuple(idxs)

def find_vars(expr):
    if (isinstance(expr, IndexExpr)):
        return [z for x in expr.indices for z in find_vars(x)]
    if (isinstance(expr, int)):
        return []
    elif (isinstance(expr, float)):
        return []
    if (isinstance(expr, IntConst)):
        return []
    elif (isinstance(expr, FloatConst)):
        return []
    elif (isinstance(expr, BoolConst)):
        return []
    elif (isinstance(expr, Ref)):
        return [expr.name]
    elif (isinstance(expr, BinOp)):
        left = find_vars(expr.left)
        right = find_vars(expr.right)
        return left + right
    elif (isinstance(expr, CmpOp)):
        left = find_vars(expr.left)
        right = find_vars(expr.right)
        return left + right
    elif (isinstance(expr, UnOp)):
        return find_vars(expr.e)
    elif (isinstance(expr, Mfunc)):
        return find_vars(expr.e)
    else:
        raise Exception(f"Unknown type for {expr}, {type(expr)}")

def eval_expr(expr, scope, dummify=False):
    if (isinstance(expr, sympy.Basic)):
        return expr
    elif (isinstance(expr, int)):
        return expr
    elif (isinstance(expr, float)):
        return expr
    if (isinstance(expr, IntConst)):
        return expr.val
    elif (isinstance(expr, FloatConst)):
        return expr.val
    elif (isinstance(expr, BoolConst)):
        return expr.val
    elif (isinstance(expr, str)):
        ref_val = scope_lookup(expr, scope)
        return eval_expr(ref_val, scope, dummify=dummify)
    elif (isinstance(expr, Ref)):
        ref_val = scope_lookup(expr.name, scope)
        return eval_expr(ref_val, scope, dummify=dummify)
    elif (isinstance(expr, BinOp)):
        left = eval_expr(expr.left, scope, dummify=dummify)
        right = eval_expr(expr.right, scope, dummify=dummify)
        return op_table[expr.op](left, right)
    elif (isinstance(expr, CmpOp)):
        left = eval_expr(expr.left, scope, dummify=dummify)
        right = eval_expr(expr.right, scope, dummify=dummify)
        return op_table[expr.op](left, right)
    elif (isinstance(expr, UnOp)):
        e = eval_expr(expr.e, scope, dummify=dummify)
        return op_table[expr.op](e)
    elif (isinstance(expr, Mfunc)):
        e = eval_expr(expr.e, scope, dummify=dummify)
        return op_table[expr.op](e)
    elif (isinstance(expr, RangeVar)):
        if (not dummify):
            raise Exception(f"Range variable {expr} cannot be evaluated directly, please specify a specific variable  or pass in dummify=True")
        else:
            return sympy.Symbol(expr.var)
    else:
        raise Exception(f"unsupported expr type {type(expr)}")


def eval_remote_call(r_call_with_scope, **kwargs):
    r_call = r_call_with_scope.remote_call
    compute = r_call.compute
    scope = r_call_with_scope.scope.copy()
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
        print("Out indices ", indices)
        op = lp.RemoteWrite(i + num_args, matrix, compute_instr.results[i], i, *indices)
        outputs.append(op)
    read_instrs =  [x for x in pyarg_list if isinstance(x, lp.RemoteRead)]
    write_instrs = outputs
    return lp.InstructionBlock(read_instrs + [compute_instr] + write_instrs)

def qr_null(A, tol=None):
    """Computes the null space of A using a rank-revealing QR decomposition"""
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()

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

def extract_constant(expr):
    consts = [term for term in expr.args if not term.free_symbols]
    if (len(consts) == 0):
        return sympy.Integer(0)
    const = expr.func(*consts)
    return const





def enumerate_possibilities(idxs, rhs, var_names, scope):
    var_limits = []
    for var in var_names:
        range_var = scope_lookup(str(var), scope)
        assert isinstance(range_var, RangeVar)
        start = eval_expr(range_var.start, scope, dummify=True)
        end = eval_expr(range_var.end, scope, dummify=True)
        step = eval_expr(range_var.step, scope, dummify=True)
        var_limits.append((start,end,step))


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

    def brute_force_limits(sol, scope, var_names, var_limits, conds):
        simple_var = None
        simple_var_idx = None
        print("in brute force")
        var_limits = [(eval_expr(low, scope), eval_expr(high, scope), eval_expr(step, scope)) for (low, high, step) in var_limits]
        lows = []
        highs = []
        for (i,(var_name, (low,high,step))) in enumerate(zip(var_names, var_limits)):
            if (not isinstance(low, int)):
                low = low.subs(scope)
            if (not isinstance(high, int)):
                high = high.subs(scope)
            if (not isinstance(step, int)):
                step = step.subs(scope)
            lows.append(low)
            highs.append(high)
            if is_constant(low) and is_constant(high) and is_constant(step):
                simple_var = var_name
                simple_var_idx = i
                print("svar", simple_var, low, high)
                break
        if (simple_var is None):
            raise Exception("NO simple var in loop lows: {0}, highs: {1} ".format(lows, highs))
        limits = var_limits[simple_var_idx]
        solutions = []
        if ((sol[0].is_constant()) and (sol[0] < limits[0] or sol[0] >= limits[1])):
                return []
        simple_var_func = sympy.lambdify(simple_var, sol[0], "numpy")
        print("limits", limits)
        for val in range(limits[0], limits[1], limits[2]):
            var_values = var_values.copy()
            var_values[str(simple_var)] = int(sol[0].subs({simple_var: val}).subs(var_values))
            if(var_values[str(simple_var)] != val):
                continue
            limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
            if (len(limits_left) > 0):
                var_names_recurse = [x for x in var_names if x != simple_var]
                solutions += brute_force_limits(sol[1:], var_values, var_names_recurse, limits_left, conds)
            else:
                solutions.append(var_values)
        return [dict(zip(var_names, sol)) for sol in solutions]
    if (len(var_names) == 0): return []
    # brute force solve in_ref vs out_ref
    linear_eqns = [idx - val for idx,val in
                   zip(idxs, rhs)]
    #TODO sort var_names by least constrainted
    solve_vars  = []
    simple_var = None
    simple_var_idx = None
    #print("var names", var_names)
    #print("var limits", var_limits)
    assert len(var_names) == len(var_limits)
    for (i,(var_name, (low,high,step))) in enumerate(zip(var_names, var_limits)):
        if (is_constant(low) and is_constant(high) and is_constant(step)):
            simple_var = var_name
            simple_var_idx = i
            #print("simple var ", simple_var, low, high)
            break

    if (simple_var is None):
        raise Exception("NO simple var in loop: limits={0}, inref={1}, outref={2}".format(list(zip(var_names, var_limits)), in_ref, out_expr))
    solve_vars = [simple_var] + [x for x in var_names if x != simple_var]
    #var_limits.insert(0, var_limits.pop(simple_var_idx))
    for i,x in enumerate(var_names):
        if (isinstance(x, str)):
            solve_vars[i] = sympy.Symbol(x)

    nonlinear = False
    linear_eqns_fixed = []
    non_lin_idxs = []
    nl_map = {}
    for i, eq in enumerate(linear_eqns):
        curr_nl = (not is_linear(eq, solve_vars))
        nonlinear = nonlinear or curr_nl
        if (curr_nl):
            placeholder = sympy.Symbol('__nl{0}__'.format(i))
            non_lin_idxs.append(len(solve_vars))
            solve_vars.append(placeholder)
            linear_eqns_fixed.append(placeholder)
            nl_map[placeholder] = eq
        else:
            linear_eqns_fixed.append(eq)
    t = time.time()
    sols = list(sympy.linsolve(linear_eqns_fixed, solve_vars))
    print("Linear sols", sols, "solve_vars", solve_vars)
    e = time.time()
    good_sols = sols
    if (nonlinear):
       # print(f"Non linear solution to eqns: {linear_eqns_fixed} is {sols}, solve vars: {solve_vars}, nl_idxs is {non_lin_idxs}")
        good_sols = []
        for j, sol in enumerate(sols):
            lin_subs = {}
            sol = list(sol)
            for i, var in enumerate(sol):
                if (i in non_lin_idxs):
                    continue
                lin_subs[solve_vars[i]] = var
            bad_sol = False
            for i in non_lin_idxs:
                assert sol[i] == 0
                before = sol[i]
                nl_var = solve_vars[i].subs(nl_map)
                nl_sub = nl_var.subs(lin_subs)
                solve_vars[i] = nl_sub
                del sol[i]
                if (is_constant(nl_sub) and nl_sub != 0):
                    bad_sol = True
                    break

            if (not bad_sol):
                good_sols.append(sol)
        sols = good_sols
    if (len(sols) > 0):
        print("Non LINEAR SOLS are ", dict(zip(solve_vars, sols[0])))
    return [(0,0,0)]
    print("all varnames", var_names)
    conds = []
    #return [dict(zip(var_names, sol)) for sol in sols]
    # three cases
    # case 1 len(sols) == 0 -> no solution
    # case 2 exact (single) solution and solution is integer
    # case 3 parametric form -> enumerate
    if (len(sols) == 0):
        return []
    elif(len(sols) == 1):
        sol = sols[0]
        if (np.all([x.is_constant() for x in sol]) and
                np.all([x.is_Integer == True for x in sol])):
            solutions = [dict(zip([str(x) for x in solve_vars], sol))]
        else:
            limits = var_limits[simple_var_idx]
            #print("enumerating through", simple_var)
            #print("simple_var_limits", limits)
            solutions = []
            #print("simple_var, sol", simple_var, sol)
            simple_var_func = sympy.lambdify(simple_var, sol[0], "numpy")
            if ((not sol[0].is_Symbol) and (sol[0] < limits[0] or sol[0] >= limits[1])):
                return []
            for val in range(limits[0], limits[1], limits[2]):
                var_values = {}
                print('val', simple_var_func(val))
                var_values[str(simple_var)] = int(simple_var_func(val))
                if(var_values[str(simple_var)] != val):
                    continue
                limits_left = [x for (i,x) in enumerate(var_limits) if i != simple_var_idx]
                if (len(limits_left) > 0):
                    var_names_recurse = [x for x in var_names if x != simple_var]
                    scope = scope.copy()
                    scope.update(var_values)
                    #print("Length var names recurse ", len(var_names_recurse))
                    #print("Length limits ", len(limits_left))
                    solutions += brute_force_limits(sol[1:], scope, var_names_recurse, limits_left, conds)
                else:
                    solutions.append(var_values)

    else:
        raise Exception("Linear equaitons should have 0, 1 or infinite solutions: {0}".format(sols))
    ret_solutions = []
    conds = []
    for sol in solutions:
        bad_sol = False
        resolved, val = check_cond(conds, sol)
        if (not resolved):
            raise Exception("Unresolved conditional conds={0}, var_values={1}".format(sol, conds))

        bad_sol = (bad_sol) or (not val)

        for (var_name, (low,high,step)) in zip(var_names, var_limits):
            if (not (isinstance(low, Number))):
                low = low.subs(sol)
            if (not (isinstance(high, Number))):
                high = high.subs(sol)
            if sol[str(var_name)] < low or sol[str(var_name)] >= high:
                bad_sol = True

        if (not bad_sol):
            ret_solutions.append(sol)
    e = time.time()
    ret = utils.remove_duplicates(ret_solutions)
    return ret


def dependency_solve(solve_vars, linear_lhs, linear_rhs, nonlinear_lhs, nonlinear_rhs, orig_scope, mod_scope, tol=1e-8):

    ''' Solve lhs(x) = rhs  for x
        if lhs is nonlinear/undetermined
         use orig_scope for program enumeration
    '''
    A = sympy.Matrix(linear_lhs)
    b = sympy.Matrix(linear_rhs)
    assert(len(b.shape) == 1 or b.shape[1] == 1)
    rank_left = A.shape[1]
    sols = []
    sol_stack = [(A, b, mod_scope.copy(),  solve_vars.copy(), {})]
    while (len(sol_stack) > 0):
        A_local, b_local, local_mod_scope, local_solve_vars, partial_sol = sol_stack.pop(0)
        if (np.product(A_local.shape) == 0):
            sols.append(partial_sol)
            continue
        x = sympy.linsolve((A_local, b_local), [sympy.Symbol(x) for x in solve_vars])
        # if not 0 -> 1
        assert len(x) == 1
        x = list(x)[0]
        if (len(x) == 0): continue
        bad_sol = False
        uq = np.all([is_constant(_) for _ in x])

        if (uq):
            assert np.all([is_constant(_) for _ in x])
            for svar,val in zip(local_solve_vars, x):
                if (int(val) != val):
                    bad_sol = True
                    break
                partial_sol[str(svar)] = int(val)
            if (bad_sol): continue
            sols.append(partial_sol)
        else:
            to_remove = []
            cols = []
            for i, (svar, val) in enumerate(zip(local_solve_vars, x)):
                if is_constant(val):
                    partial_sol[str(svar)] = int(val)
                    local_mod_scope[str(svar)] = int(val)
                    to_remove.append(str(svar))
                else:
                    cols.append(A_local[:, i])

            A_local = sympy.Matrix.vstack(*cols)
            for r in to_remove:
                local_solve_vars.remove(r)
            print("partial sol", partial_sol)

            # rank deficient solution
            range_obj = None
            simple_var = None
            simple_idx = None
            min_range = float('inf')

            for i,s in enumerate(local_solve_vars):
                range_var = scope_lookup(str(s), orig_scope)
                print("range_var", range_var)
                print("val", s)
                assert isinstance(range_var, RangeVar)
                start = eval_expr(range_var.start, local_mod_scope)
                end = eval_expr(range_var.end, local_mod_scope)
                step = eval_expr(range_var.step, local_mod_scope)
                print(start, end, step)
                try:
                    start = int(start)
                    end = int(end)
                    step = int(step)
                    print("start, end, step", start, end, step)
                    if (((end - start)/step) < min_range):
                        range_obj = range(int(start), int(end), int(step))
                        simple_var = s
                        simple_idx = i
                        local_solve_vars.remove(s)
                except ValueError:
                    pass

            print("Enumerating through..", simple_var)
            #print("Found simple var range ",  range_obj)
            if (range_obj is None): raise Exception("No valid range found")
            assert simple_var is not None
            assert simple_idx is not None
            nlr_switches = [0 for _ in nonlinear_lhs]
            for r in range_obj:
                partial_sol_local = partial_sol.copy()
                partial_sol_local[str(simple_var)] = r
                A_new = A_local.copy()
                b_new = b_local.copy()
                for i, nlr in enumerate(nonlinear_lhs):
                    print("nlr before", nlr)
                    nlr = nlr.subs(partial_sol_local)
                    print("nlr after", nlr)
                    if (nlr_switches[i] == 0):
                        still_nlr = not is_linear(nlr, solve_vars)
                        # False -> -1
                        # True -> 1
                        nlr_switches[i] = int(still_nlr)*2 - 1
                    if (nlr_switches[i] == -1):
                        new_row = []
                        for var in local_solve_vars:
                            new_row.append(nlr.coeff(var))
                        new_row.insert(0, simple_idx)
                        new_row_rhs  = sympy.Matrix([nonlinear_rhs[i]])
                        sympy.Matrix.vstack(b_local, new_row_rhs)
                        A_new = sympy.Matrix.vstack(A_new, sympy.Matrix(new_row).T)
                        b_new = sympy.Matrix.vstack(b_new, new_row_rhs)

                new_solve_vars = local_solve_vars.copy()
                b_new  = b_new - r*A_new[:, simple_idx]
                print("old_shape", A_new.shape)
                A_new = sympy.Matrix.hstack(A_new[:, :simple_idx], A_new[:, simple_idx+1:])
                print("new_shape", A_new.shape)
                local_mod_scope_copy = local_mod_scope.copy()
                local_mod_scope_copy[str(simple_var)] = r
                sol_stack.append((A_new, b_new, local_mod_scope_copy, new_solve_vars, partial_sol_local))
    inbound_sols = []
    for sol in sols:
        for k,v in sol.items():
            range_var = scope_lookup(str(k), orig_scope)
            assert isinstance(range_var, RangeVar)
            local_mod_scope = orig_scope.copy()
            local_mod_scope.update(sol)
            start = eval_expr(range_var.start, local_mod_scope)
            end = eval_expr(range_var.end, local_mod_scope)
            step = eval_expr(range_var.step, local_mod_scope)
            start = int(start)
            end = int(end)
            step = int(step)

            assert (isinstance(start, Number) and isinstance(end, Number) and isinstance(step, Number))
            assert int(start) == start and int(end) == end and int(step) == step
            start = int(start)
            end = int(end)
            step = int(step)
            range_obj = range(start, end, step)
            if v not in range_obj:
                bad_sol = True
                break
        for lhs, rhs in zip(nonlinear_lhs, nonlinear_rhs):
            if (lhs.subs(sol) != rhs):
                bad_sol = True
                break
        if (bad_sol):
            sols.remove(sol)
    return utils.remove_duplicates(sols)

def find_children(r_call, program, **kwargs):
    ''' Given a specific r_call and arguments to evaluate it completely
        return all other program locations that read from the output of r_call
    '''
    ib = eval_remote_call(r_call, **kwargs)
    children = []
    for inst in ib.instrs:
        if (not isinstance(inst, lp.RemoteWrite)): continue
        assert(isinstance(inst, lp.RemoteWrite))
        rhs = inst.bidxs
        for p_idx in program.keys():
            print("current p_idx", p_idx)
            r_call_abstract_with_scope = program[p_idx]
            r_call_abstract = r_call_abstract_with_scope.remote_call
            scope = r_call_abstract_with_scope.scope.copy()
            scope_orig = r_call_abstract_with_scope.scope.copy()
            for i, _arg in enumerate(r_call_abstract.args):
                if (not isinstance(_arg, IndexExpr)): continue
                assert(isinstance(_arg, IndexExpr))
                vars_for_arg = list(set(find_vars(_arg)))
                for v in vars_for_arg:
                    if not(isinstance(scope_lookup(v, scope_orig), RangeVar)):
                        vars_for_arg.remove(v)

                for v in vars_for_arg:
                    scope[str(v)] = sympy.Symbol(v)

                matrix, indices = eval_index_expr(_arg, scope)
                if (matrix != inst.matrix): continue
                # If the matrices are equal then they must
                # have same number of index variables
                assert len(rhs) == len(indices)
                loop_data = []
                types = [x.type for x in _arg.indices]
                linear_lhs = []
                linear_rhs = []
                nonlinear_lhs = []
                nonlinear_rhs = []
                local_children = enumerate_possibilities(indices, rhs, vars_for_arg, scope_orig)
                #print("children for ", inst,  "are", local_children, p_idx)
                children += local_children
                #print("Linear LHS", linear_lhs)
                #print("Linear RHS", linear_rhs)
                #print("NonLinear LHS", nonlinear_lhs)
                #print("NonLinear RHS", nonlinear_rhs)
    return utils.remove_duplicates(children)





















if __name__ == "__main__":
    N = 32
    nb = 8
    X = np.random.randn(N,N)
    I = BigMatrix("TSQR_input", shape=(int(N),int(N)), shard_sizes=(nb, nb))
    shard_matrix(I, X)
    Q = BigMatrix("TSQR_output_Q", shape=(int(N),int(N)), shard_sizes=(nb, nb))
    R = BigMatrix("TSQR_output_R", shape=(int(N),int(N)), shard_sizes=(nb, nb))
    program = lpcompile(TSQR)(I, Q, R, int(np.ceil(N/nb)))
    print(program)
    starters = program.starters
    print("STARTER", starters[1])
    print("TERMINATORS", program.find_terminators())
    c = program.get_children(*starters[1])
    print("starter children", c)
    c2 = program.get_children(*c[0])
    print("starter children 2", c2)


    operator_expr = program.get_expr(c[0][0])
    inst_block = operator_expr.eval_operator(c[0][1])

    operator_expr = program.get_expr(c2[0][0])
    inst_block = operator_expr.eval_operator(c2[0][1])



