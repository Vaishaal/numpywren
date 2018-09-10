import numpy as np
import unittest
import pywren
from numpywren import kernels

class KernelsTest(unittest.TestCase):
    def test_kernel_qr(self):
        N = 32
        X =  np.random.randn(N, N)
        f = lambda _: kernels.fast_qr(X[:4, :4])
        f(0)
        f1 = lambda _: kernels.slow_qr(X[:4, :4])
        f1(0)

    def test_kernel_qr_pywren(self):
        N = 32
        X =  np.random.randn(N, N)
        f0 = lambda _: kernels.fast_qr(X[:4, :4])
        f1 = lambda _: kernels.slow_qr(X[:4, :4])
        pwex = pywren.default_executor()
        futures = pwex.map(f1, [0], exclude_modules=["site-packages"])
        print(futures[0].result())
        futures = pwex.map(f0, [0], exclude_modules=["site-packages"])
        print(futures[0].result())


if __name__ == "__main__":
    k_test = KernelsTest()
    k_test.test_kernel_qr()


