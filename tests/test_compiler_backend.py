import unittest

import numpy as np
import sympy
from sympy import sympify

import numpywren.lambdapack as lp
from numpywren.matrix import BigMatrix
from numpywren.matrix_init import shard_matrix


class CompilerBackendTestClass(unittest.TestCase):
    def test_single_operator(self):
        X_sharded = BigMatrix("test_0", shape=[4, 4], shard_sizes=[2, 2])
        program = lp.Program([
            lp.CholeskyExpr((X_sharded, (sympify(1), sympify(0))),
                            (X_sharded, (sympify(1), sympify(0))))
        ])
        chol = program.get_expr(0)
        assert program.num_exprs == 1
        assert chol.num_exprs == 1
        assert chol.eval_read_refs({}) == [(X_sharded, (1, 0))]
        assert chol.eval_write_ref({}) == (X_sharded, (1, 0))
        assert program.eval_read_operators((X_sharded, (sympify(0), sympify(0)))) == []
        assert program.eval_read_operators((X_sharded, (sympify(1), sympify(0)))) == [(0, {})]
        X_sharded.free()

    def test_single_for_loop(self):
        X_sharded = BigMatrix("test_0", shape=[4, 4], shard_sizes=[2, 2])
        i = sympify("i")
        program = lp.Program([
            lp.For(var=i, limits=[sympify(0), sympify(2)], body=[
                lp.CholeskyExpr((X_sharded, (sympify(0), i)),
                                (X_sharded, (i, sympify(0)))),
                lp.CholeskyExpr((X_sharded, (i, sympify(1))),
                                (X_sharded, (sympify(1), i))),
            ])
        ])
        chol1 = program.get_expr(0)
        chol2 = program.get_expr(1)
        assert program.num_exprs == 2
        assert chol1.eval_read_refs({i: 1}) == [(X_sharded, (0, 1))]
        assert chol2.eval_write_ref({i: 1}) == (X_sharded, (1, 1))
        assert program.eval_read_operators((X_sharded, (sympify(1), sympify(1)))) == [(1, {i: 1})]
        assert program.eval_read_operators((X_sharded, (sympify(2), sympify(1)))) == []
        assert (sorted(program.eval_read_operators((X_sharded, (sympify(0), sympify(1))))) ==
                sorted([(0, {i: 1}), (1, {i: 0})]))
