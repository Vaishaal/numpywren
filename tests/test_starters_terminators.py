import time
import random
from timeit import default_timer as timer
import string

from numpywren import compiler
from numpywren.matrix import BigMatrix
import numpy as np
from numpywren.matrix_init import shard_matrix
from numpywren.algs import *
from numpywren.compiler import lpcompile, walk_program, find_starters, find_terminators
from test_dependency_analyze import dummy_matrix

def test_cholesky():
    A = dummy_matrix()
    B = dummy_matrix()
    C = dummy_matrix(num_dims=3)
    program = lpcompile(CHOLESKY)(A,B,C,313,0)
    assert (find_starters(program, input_matrices=["I"])) == [(0, {})]
    assert len(find_terminators(program, output_matrices=["O"])) == 49141

def test_qr():
    A = dummy_matrix(num_dims=2)
    V = dummy_matrix(num_dims=2)
    T = dummy_matrix(num_dims=2)
    R = dummy_matrix(num_dims=2)
    S = dummy_matrix(num_dims=4)
    M = 256
    program = lpcompile(QR)(A,V,T,R,S,M,0)
    assert len(find_starters(program, input_matrices=["I"])) == M
    assert len(find_terminators(program, output_matrices=["Rs"])) == 129920

def test_gemm():
    A = dummy_matrix(num_dims=2)
    B = dummy_matrix(num_dims=2)
    M = 4
    N = 4
    K = 4
    Temp = dummy_matrix(num_dims=4)
    Out = dummy_matrix(num_dims=3)
    program = lpcompile(GEMM)(A,B,M,N,K,Temp,Out)
    assert len(find_starters(program, input_matrices=["A", "B"]))  == M*N*K
    assert len(find_terminators(program, output_matrices=["Out"]))  == M*N


if __name__ == "__main__":
    test_qr()


