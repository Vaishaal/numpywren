from numpywren import frontend, exceptions
import unittest
import astor
import ast
import inspect



def F1(a:int, b:int) -> int:
    return a//b

def F2(a:float, b:int) -> float:
    return a + b

def F3(a:float, b:int) -> float:
    c = a + b
    d = log(c)
    e = ceiling(d)
    return c

def F4(a:float, b:int) -> float:
    c = a + b
    d = log(c)
    if (c > d):
        e = log(d)
    else:
        e = d
    return e

def F5(a:float, b:int) -> float:
    c = a + b
    d = log(c)
    e = d**c
    if (c > d):
        f = e
    else:
        f = d
    return f

def F6(a:float, b:int, c:int) -> float:
    return ((a + b) * (b**a))/floor(c)

def F7_err(N:int, M:int) -> float:
    c = a + b
    d = log(c)
    e = d**c
    if (c > d):
        f = c
        g = e
    else:
        f = e
    return f


def F7_no_err(a:int, b:int) -> float:
    c = a + b
    d = log(c)
    e = d**c
    if (c > d):
        f = d
    else:
        f = e
    return d


def F8(N:int, M:int):
    for i in range(N):
        for j in range(i+1, M):
            if (i  < j/2):
                z = i + j
            else:
                z = i - j

def F9(N:int, M:int):
    for i in range(N):
        for j in range(i+1, M):
            if (i  < j/2):
                if (j > log(M)):
                    z = i + j
                else:
                    z = 2*i+4*j
            else:
                z = i - j


class FrontEndTest(unittest.TestCase):
    def test_types_simple(self):
        parser, type_checker, f2_ast = frontend.lpcompile(F2)
        tree = astor.dump_tree(f2_ast)
        assert type_checker.decl_types['a'] == frontend.ConstFloatType
        assert type_checker.decl_types['b'] == frontend.ConstIntType

    def test_types_simple_2(self):
        parser, type_checker, f3_ast = frontend.lpcompile(F3)
        tree = astor.dump_tree(f3_ast)
        assert type_checker.decl_types['c'] == frontend.ConstFloatType
        assert type_checker.decl_types['d'] == frontend.ConstFloatType
        assert type_checker.decl_types['e'] == frontend.ConstIntType

    def test_types_simple_if(self):
        parser, type_checker, f_ast = frontend.lpcompile(F4)
        tree = astor.dump_tree(f_ast)
        assert type_checker.decl_types['c'] == frontend.ConstFloatType
        assert type_checker.decl_types['d'] == frontend.ConstFloatType
        assert type_checker.decl_types['e'] == frontend.ConstFloatType

    def test_types_compound_expr_3(self):
        parser, type_checker, f_ast = frontend.lpcompile(F6)
        assert type_checker.return_node_type == frontend.ConstFloatType

    def test_types_if_statement_err(self):
        try:
            parser, type_checker, f_ast = frontend.lpcompile(F7_err)
        except exceptions.LambdaPackParsingException:
            pass

    def test_types_if_statement_no_err(self):
        parser, type_checker, f_ast = frontend.lpcompile(F7_no_err)
        assert type_checker.decl_types['f'] == frontend.ConstFloatType
        assert type_checker.decl_types['d'] == frontend.ConstFloatType

    def test_types_for_loop_if_statment(self):
        parser, type_checker, f_ast = frontend.lpcompile(F8)
        assert type_checker.decl_types['z'] == frontend.LinearIntType
        assert type_checker.decl_types['i'] == frontend.LinearIntType

    def test_types_for_loop_nested_if_statment(self):
        parser, type_checker, f_ast = frontend.lpcompile(F9)
        assert type_checker.decl_types['z'] == frontend.LinearIntType
        assert type_checker.decl_types['i'] == frontend.LinearIntType













