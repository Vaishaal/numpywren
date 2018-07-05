import unittest

import numpy as np
import sympy
from sympy import sympify

import numpywren.lambdapack as lp
from numpywren.lambdapack import RemoteCholesky, RemoteTRSM, RemoteSYRK, RemoteRead, RemoteWrite, InstructionBlock, RemoteReturn
from numpywren.matrix import BigMatrix
from numpywren import compiler
from numpywren.matrix_init import shard_matrix


class CompilerBackendTestClass(unittest.TestCase):
    def test_single_operator(self):
        X_sharded = BigMatrix("test_0", shape=[4, 4], shard_sizes=[2, 2])
        program = compiler.Program("test", [
            compiler.OperatorExpr(0, RemoteCholesky, [(X_sharded, (sympify(1), sympify(0)))],
                            (X_sharded, (sympify(1), sympify(0))), is_output=False, is_input=False)
        ])
        chol = program.get_expr(0)
        assert program.num_exprs == 1
        assert chol.num_exprs == 1
        assert chol.eval_read_refs({}) == [(X_sharded, (1, 0))]
        assert chol.eval_write_ref({}) == (X_sharded, (1, 0))
        assert program.eval_read_operators((X_sharded, (0, 0))) == []
        assert program.eval_read_operators((X_sharded, (1, 0))) == [(0, {})]
        assert program.eval_write_operators((X_sharded, (1, 0))) == [(0, {})]
        assert program.eval_write_operators((X_sharded, (1, 1))) == []
        X_sharded.free()

    def test_single_for_loop(self):
        X_sharded = BigMatrix("test_1", shape=[4, 4], shard_sizes=[2, 2])
        i = sympy.Symbol("i")
        program = compiler.Program("test_for",[
            compiler.BackendFor(var=i, limits=[sympify(0), sympify(2)], body=[
                compiler.OperatorExpr(0, RemoteCholesky, [(X_sharded, (sympify(0), i))],
                                (X_sharded, (i, sympify(0)))),
                compiler.OperatorExpr(0, RemoteCholesky, [(X_sharded, (i, sympify(1)))],
                                (X_sharded, (sympify(1), i)))
            ])
        ])
        chol1 = program.get_expr(0)
        chol2 = program.get_expr(1)
        assert program.num_exprs == 2
        assert chol1.eval_read_refs({i: 1}) == [(X_sharded, (0, 1))]
        assert chol2.eval_write_ref({i: 1}) == (X_sharded, (1, 1))
        assert program.eval_read_operators((X_sharded, (sympify(1), sympify(1)))) == [(1, {i: 1})]
        assert program.eval_read_operators((X_sharded, (sympify(2), sympify(1)))) == []
        assert (self.sort_eval(program.eval_read_operators((X_sharded, (sympify(0), sympify(1))))) ==
                self.sort_eval([(0, {i: 1}), (1, {i: 0})]))
        assert (self.sort_eval(program.eval_write_operators((X_sharded, (sympify(1), sympify(1))))) ==
                self.sort_eval([(1, {i: 1})]))
        X_sharded.free()

    def test_nested_for_loop(self):
        X_sharded = BigMatrix("X_test_2", shape=[4, 4], shard_sizes=[2, 2])
        Y_sharded = BigMatrix("Y_test_2", shape=[4, 4], shard_sizes=[2, 2])
        i, j = sympy.symbols(["i", "j"])
        program = compiler.Program("nested_for", [
            compiler.BackendFor(var=i, limits=[sympify(0), sympify(2)], body=[
                compiler.OperatorExpr(0, RemoteCholesky, [(X_sharded, (i, sympify(1)))],
                                (X_sharded, (sympify(1), i))),
                compiler.BackendFor(var=j, limits=[sympify(i), sympify(2)], body=[
                    compiler.OperatorExpr(0, RemoteSYRK, [(Y_sharded, (j - i, i)),
                                (Y_sharded, (i, i)),
                                (Y_sharded, (j, sympify(1)))],
                                (X_sharded, (i, j - i))),
                ])
            ])
        ])
        chol = program.get_expr(0)
        syrk = program.get_expr(1)
        assert program.num_exprs == 2
        assert chol.eval_read_refs({i: 1, j: 0}) == [(X_sharded, (1, 1))]
        assert (sorted(syrk.eval_read_refs({i: 1, j: 1})) ==
                sorted([(Y_sharded, (0, 1)), (Y_sharded, (1, 1)), (Y_sharded, (1, 1))]))
        assert syrk.eval_write_ref({i: 0, j: 1}) == (X_sharded, (0, 1))
        assert program.eval_read_operators((X_sharded, (0, 1))) == [(0, {i: 0})]
        assert program.eval_read_operators((X_sharded, (0, 2))) == []
        assert program.eval_read_operators((Y_sharded, (0, 2))) == []
        assert (self.sort_eval(program.eval_read_operators((Y_sharded, (0, 0)))) ==
                self.sort_eval([(1, {i: 0, j: 0}), (1, {i: 0, j: 1})]))
        assert (self.sort_eval(program.eval_read_operators((Y_sharded, (1, 0)))) ==
                self.sort_eval([(1, {i: 0, j: 1})]))
        assert (self.sort_eval(program.eval_read_operators((Y_sharded, (1, 1)))) ==
                self.sort_eval([(1, {i: 0, j: 1}), (1, {i: 1, j: 1})]))
        assert (self.sort_eval(program.eval_read_operators((X_sharded, (1, 1)))) ==
                self.sort_eval([(0, {i: 1})]))
        assert (self.sort_eval(program.eval_write_operators((X_sharded, (1, 0)))) ==
                self.sort_eval([(0, {i: 0}), (1, {i: 1, j: 1})]))
        X_sharded.free()
        Y_sharded.free()

    def sort_eval(self, l):
        key = lambda item: (item[0], sorted([(key.name, value) for key, value in item[1].items()]))
        return sorted(l, key=key)

