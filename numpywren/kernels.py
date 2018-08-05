import numpy as np

def add(*args, **kwargs):
    out = np.zeros(args[0].shape)
    for a in args:
        out += a
    return out

def qr_factor(*blocks, **kwargs):
    ins = np.vstack(blocks)
    out = np.linalg.qr(ins)
    print("IN SHAPE", ins.shape)
    print("OUT Q SHAPE", out[0].shape)
    print("OUT R SHAPE", out[1].shape)
    return out

