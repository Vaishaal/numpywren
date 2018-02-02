import numpy as np
from numpywren.matrix import BigMatrix, BigSymmetricMatrix
import numpywren.matrix_utils as utils
import concurrent.futures as fs
import time
import scipy.linalg
import sklearn.metrics
from sklearn.datasets import fetch_mldata
from numpywren.matrix_init import local_numpy_init
from scipy.linalg import solve
import argparse

def backwards_solve(bigm):
    last_block = bigm._block_idxs(1)[-1]
    print("Grabbing y")
    y1 = utils.get_col(bigm, last_block, mmap_loc="/dev/shm/y1")
    y0 = utils.get_col(bigm, last_block - 1, mmap_loc="/dev/shm/y0")
    y = np.hstack((y0, y1))[:, -1000:]
    x = np.zeros(y.shape)
    executor = fs.ProcessPoolExecutor(64)
    for block_idx, (b_start, b_end) in list(zip(bigm._block_idxs(0)[::-1], bigm._blocks(0)[::-1])):
        print("block_idx", block_idx)
        t = time.time()
        futures = utils.get_matrix_blocks_full_async(bigm, "/dev/shm/block", [block_idx], list(range(block_idx, last_block+1)), big_axis=1, executor=executor, workers=64)
        fs.wait(futures)
        [f.result() for f in futures]
        A_block = utils.load_mmap(*futures[0].result())[:, :-1000]
        print(A_block.shape)
        y_block = y[b_start:b_end]
        block_size  = bigm.shard_sizes[0]
        compute_start = time.time()
        A_bb = A_block[:, :min(block_size, A_block.shape[1])]
        A_rest = A_block[:, min(block_size, A_block.shape[1]):]
        if (A_rest.shape[1] != 0):
            rhs = y_block - A_rest.dot(x[b_end:])
        else:
            rhs = y_block
        x[b_start:b_end] = scipy.linalg.solve_triangular(A_bb, rhs)
        if (block_idx % 10 == 0):
            print("Writing model")
            np.save("/dev/shm/backsub_{0}".format(bigm.key), x)
        compute_end = time.time()
        print("Compute took {0} seconds".format(compute_end - compute_start))
        e = time.time()
        print("Block {0} took {1} seconds".format(block_idx, e - t))
    np.save("/dev/shm/backsub_{0}".format(bigm.key), x)
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('matrix_key', type=str, help="LU decomposition")
    args = parser.parse_args()
    bigm  = BigMatrix(args.matrix_key, bucket="pictureweb", shape=[1281167, 1282167], shard_sizes=[4096,4096])
    backwards_solve(bigm)




