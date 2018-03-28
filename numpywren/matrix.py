import base64
import concurrent.futures as fs
import io
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import time

import boto3
import botocore
import cloudpickle
import numpy as np
import pywren.wrenconfig as wc

from . import matrix_utils
from .matrix_utils import list_all_keys, block_key_to_block, get_local_matrix, key_exists_async
import asyncio
import aiobotocore

cpu_count = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)

try:
    DEFAULT_BUCKET = wc.default()['s3']['bucket']
except Exception as e:
    DEFAULT_BUCKET = ""

class BigMatrix(object):
    """
    A multidimensional array stored in S3, sharded in blocks of a given size.

    Parameters
    ----------
    key : string
        The S3 key to store this matrix at.
    shape : tuple of int, optional
        Shape of the array. If set to None, the array with the given key
        must already exist in S3 with a valid header. 
    shard_sizes : tuple of int, optional
        Shape of the array blocks. If shape is not None this must be set,
        otherwise it will be ignored.
    bucket : string, optional
        Name of the S3 bucket where the matrix will be stored.
    prefix : string, optional
        Prefix that will be appended to the key name.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Determines
        the type of the object stored in the array.
    parent_fn : function, optional
        A function that gets called when a previously uninitialized block is
        accessed. Gets passed the BigMatrix object and the relevant block index
        and is expected to appropriately initialize the given block.
    write_header : bool, optional
        If write_header is True then a header will be stored alongside the array
        to allow other BigMatrix objects to be initialized with the same key
        and underlying S3 representation.

    Notes
    -----
    BigMatrices deal with two types of indexing. Absolute and block indexing.
    Absolute indexing is simply the standard method of indexing arrays by their
    elements while block indexing accesses whole blocks.
    """
    def __init__(self,
                 key,
                 shape=None,
                 shard_sizes=None,
                 bucket=DEFAULT_BUCKET,
                 prefix='numpywren.objects/',
                 dtype=np.float64,
                 parent_fn=None,
                 write_header=False):
        if bucket is None:
            bucket = os.environ.get('PYWREN_LINALG_BUCKET')
            if bucket is None:
                raise Exception("Bucket not provided and environment variable " +
                                "PYWREN_LINALG_BUCKET not provided.")
        self.bucket = bucket
        self.prefix = prefix
        self.key = key
        self.key_base = os.path.join(prefix, self.key)
        self.dtype = dtype
        self.parent_fn = parent_fn
        if (shape == None or shard_sizes == None):
            header = self.__read_header__()
        else:
            header = None
        if header is None and shape is None:
            raise Exception("Header doesn't exist and no shape provided.")
        elif shape is None:
            # Initialize the matrix parameters from S3.
            self.shard_sizes = header['shard_sizes']
            self.shape = header['shape']
            self.dtype = self.__decode_dtype__(header['dtype'])
        else:
            # Initialize the matrix parameters from inputs.
            self.shape = shape
            self.shard_sizes = shard_sizes
            self.dtype = dtype

        if (self.shard_sizes is None) or (len(self.shape) != len(self.shard_sizes)):
            raise Exception("shard_sizes should be same length as shape.")
        self.symmetric = False
        if write_header:
            # Write a header if you want to load this value later.
            self.__write_header__()

    @property
    def T(self):
        """Return the transpose with the same underlying representation."""
        return BigMatrixView(self, [slice(None, None, None)] * len(self.shape), transposed=True)

    @property
    def blocks_exist(self):
        """
        Return the absolute start and end indices of all initialized blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a list of tuples, where
            each tuple stores the start and end indices of the block along a
            dimension.
        """
        all_keys = list_all_keys(self.bucket, self.key_base)
        return list(filter(lambda x: x is not None, map(block_key_to_block, all_keys)))

    @property
    def blocks_not_exist(self):
        """
        Return the absolute start and end indices of all uninitialized blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a list of tuples, where
            each tuple stores the start and end indices of the block along a
            dimension.
        """
        blocks = set(self.blocks)
        block_exist = set(self.blocks_exist)
        return list(filter(lambda x: x, list(block_exist.symmetric_difference(blocks))))

    @property
    def blocks(self):
        """
        Return the absolute start and end indices of all blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a list of tuples, where
            each tuple stores the start and end indices of the block along a
            dimension.
        """
        return self._blocks()

    @property
    def block_idxs_exist(self):
        """
        Return the block indices of all initialized blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a tuple, where
            each tuple stores the block indices of the block.
        """
        all_block_idxs = self.block_idxs
        all_blocks = self.blocks
        blocks_exist = set(self.blocks_exist)
        block_idxs_exist = []
        for i, block in enumerate(all_blocks):
            if block in blocks_exist:
                block_idxs_exist.append(all_block_idxs[i])
        return block_idxs_exist

    @property
    def block_idxs_not_exist(self):
        """
        Return the block indices of all uninitialized blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a tuple, where
            each tuple stores the block indices of the block.
        """
        block_idxs = set(self.block_idxs)
        block_idxs_exist = set(self.block_idxs_exist)
        return list(filter(lambda x: x, list(block_idxs_exist.symmetric_difference(block_idxs))))

    @property
    def block_idxs(self):
        """
        Return the block indices of all blocks.

        Returns
        -------
        blocks : array_like of int
            A list of block indices, where each block is represented by one
            element in the list. Each block is itself a tuple, where
            each tuple stores the block indices of the block.
        """
        return self._block_idxs()

    def get_block(self, *block_idx):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        get_block_async_coro = self.get_block_async(loop, *block_idx)
        res = loop.run_until_complete(asyncio.ensure_future(get_block_async_coro))
        return res

    async def get_block_async(self, loop, *block_idx):
        """
        Given a block index, get the contents of the block.

        Parameters
        ----------
        block_idx : int or sequence of ints
            The index of the block to retrieve.

        Returns
        -------
        block : ndarray
            The block at the given index as a numpy array.
        """
        if (loop == None):
            loop = asyncio.get_event_loop()

        if (len(block_idx) != len(self.shape)):
            raise Exception("Get block query does not match shape")
        key = self.__shard_idx_to_key__(block_idx)
        exists = await key_exists_async(self.bucket, key, loop)
        if (not exists and self.parent_fn == None):
            print(self.bucket)
            print(key)
            raise Exception("Key does not exist, and no parent function prescripted")
        elif (not exists and self.parent_fn != None):
            X_block = self.parent_fn(self, *block_idx)
        else:
            bio = await self.__s3_key_to_byte_io__(key, loop=loop)
            X_block = np.load(bio).astype(self.dtype)
        return X_block

    def put_block(self, block, *block_idx):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        put_block_async_coro = self.put_block_async(block, loop, *block_idx)
        res = loop.run_until_complete(asyncio.ensure_future(put_block_async_coro))
        return res

    async def put_block_async(self, block, loop=None, *block_idx):
        """
        Given a block index, sets the contents of the block.

        Parameters
        ----------
        block : ndarray
            The array to set the block to.
        block_idx : int or sequence of ints
            The index of the block to set.

        Returns
        -------
        response : dict
            The response from S3 containing information on the status of
            the put request.

        Notes
        -----
        For details on the S3 response format see:
        http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.put_object
        """
        if (loop == None):
            loop = asyncio.get_event_loop()

        real_idxs = self.__block_idx_to_real_idx__(block_idx)
        current_shape = tuple([e - s for s,e in real_idxs])

        if (block.shape != current_shape):
            raise Exception("Incompatible block size: {0} vs {1}".format(block.shape, current_shape))

        block = block.astype(self.dtype)
        key = self.__shard_idx_to_key__(block_idx)
        return await self.__save_matrix_to_s3__(block, key, loop)

    def delete_block(self, *block_idx):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        delete_block_async_coro = self.delete_block_async(loop, *block_idx)
        res = loop.run_until_complete(asyncio.ensure_future(delete_block_async_coro))
        return res

    async def delete_block_async(self, loop=None, *block_idx):
        """
        Delete the block at the given block index.

        Parameters
        ----------
        block_idx : int or sequence of ints
            The index of the block to delete.

        Returns
        -------
        response : dict
            The response from S3 containing information on the status of
            the delete request.

        Notes
        -----
        For details on the S3 response format see:
        http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.delete_object
        """
        if (loop == None):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        key = self.__shard_idx_to_key__(block_idx)
        session = aiobotocore.get_session(loop=loop)
        async with session.create_client('s3', use_ssl=False, verify=False, region_name="us-west-2") as client:
            resp = client.delete_object(Key=key, Bucket=self.bucket)
        return resp

    def free(self):
        """Delete all allocated blocks while leaving the matrix metadata intact."""
        [self.delete_block(*x) for x in self.block_idxs_exist]
        return 0

    def delete(self):
        """Completely remove the matrix from S3."""
        self.free()
        self.__delete_header__()
        return 0

    def numpy(self, workers=cpu_count):
        """
        Convert the BigMatrix to a local numpy array.

        Parameters
        ----------
        workers : int, optional
            The number of local workers to use when converting the array.

        Returns
        -------
        out : ndarray
            The numpy version of the BigMatrix object.
        """
        return matrix_utils.get_local_matrix(self, workers)

    def _blocks(self, axis=None):
        all_blocks = []
        for i in range(len(self.shape)):
            blocks_axis = [(j, j + self.shard_sizes[i]) for j in
                           range(0, self.shape[i], self.shard_sizes[i])]
            if blocks_axis[-1][1] > self.shape[i]:
                blocks_axis.pop()

            if blocks_axis[-1][1] < self.shape[i]:
                blocks_axis.append((blocks_axis[-1][1], self.shape[i]))
            all_blocks.append(blocks_axis)

        if axis is None:
            return list(itertools.product(*all_blocks))
        elif type(axis) is not int:
            raise Exception("Axis must be an integer.")
        else:
            return all_blocks[axis]

    def _register_parent(self, parent_fn):
        self.parent_fn = parent_fn

    def _block_idxs(self, axis=None):
        idxs = [list(range(len(self._blocks(axis=i)))) for i in range(len(self.shape))]
        if axis is None:
            return list(itertools.product(*idxs))
        elif (type(axis) != int):
            raise Exception("Axis must be integer")
        else:
            return idxs[axis]

    def __get_matrix_shard_key__(self, real_idxs):
            key_string = ""

            shard_sizes = self.shard_sizes
            for ((sidx, eidx), shard_size) in zip(real_idxs, shard_sizes):
                key_string += "{0}_{1}_{2}_".format(sidx, eidx, shard_size)

            return os.path.join(self.key_base, key_string)

    def __read_header__(self):
        client = boto3.client('s3')
        try:
            key = os.path.join(self.key_base, "header")
            header = json.loads(client.get_object(Bucket=self.bucket,
                                                  Key=key)['Body'].read().decode('utf-8'))
        except Exception as e:
            header = None
        return header

    def __delete_header__(self):
        key = os.path.join(self.key_base, "header")
        client = boto3.client('s3')
        client.delete_object(Bucket=self.bucket, Key=key)

    def __block_idx_to_real_idx__(self, block_idx):
        starts = []
        ends = []
        for i in range(len(self.shape)):
            start = block_idx[i]*self.shard_sizes[i]
            end = min(start+self.shard_sizes[i], self.shape[i])
            starts.append(start)
            ends.append(end)
        return tuple(zip(starts, ends))

    def __shard_idx_to_key__(self, block_idx):

        real_idxs = self.__block_idx_to_real_idx__(block_idx)
        key = self.__get_matrix_shard_key__(real_idxs)
        return key

    async def __s3_key_to_byte_io__(self, key, loop=None):
        if (loop == None):
            loop = asyncio.get_event_loop()

        session = aiobotocore.get_session(loop=loop)
        async with session.create_client('s3', use_ssl=False, verify=False, region_name="us-west-2") as client:
            n_tries = 0
            max_n_tries = 5
            bio = None
            while bio is None and n_tries <= max_n_tries:
                try:
                    resp = await client.get_object(Bucket=self.bucket, Key=key)
                    async with resp['Body'] as stream:
                        matrix_bytes = await stream.read()
                    bio = io.BytesIO(matrix_bytes)
                except Exception as e:
                    raise
                    n_tries += 1
        if bio is None:
            raise Exception("S3 Read Failed")
        return bio

    async def __save_matrix_to_s3__(self, X, out_key, loop, client=None):
        if (loop == None):
            loop = asyncio.get_event_loop()

        session = aiobotocore.get_session(loop=loop)
        async with session.create_client('s3', use_ssl=False, verify=False, region_name="us-west-2") as client:
            outb = io.BytesIO()
            np.save(outb, X)
            response = await client.put_object(Key=out_key,
                                         Bucket=self.bucket,
                                         Body=outb.getvalue(),
                                         ACL="bucket-owner-full-control")
        return response

    def __write_header__(self):
        key = os.path.join(self.key_base, "header")
        client = boto3.client('s3')
        header = {}
        header['shape'] = self.shape
        header['shard_sizes'] = self.shard_sizes
        header['dtype'] = self.__encode_dtype__(self.dtype)
        client.put_object(Key=key,
                          Bucket=self.bucket,
                          Body=json.dumps(header),
                          ACL="bucket-owner-full-control")

    def __encode_dtype__(self, dtype):
        dtype_pickle = pickle.dumps(dtype)
        b64_str = base64.b64encode(dtype_pickle).decode('utf-8')
        return b64_str

    def __decode_dtype__(self, dtype_enc):
        dtype_bytes = base64.b64decode(dtype_enc)
        dtype = pickle.loads(dtype_bytes)
        return dtype

    def __getitem__(self, bidxs):
        return BigMatrixView(self, bidxs)

    def __str__(self):
        rep = "{0}({1})".format(self.__class__.__name__, self.key)
        return rep


class BigMatrixView(BigMatrix):
    def __init__(self, parent, parent_slices, transposed=False):
        self.parent = parent
        self.transposed = transposed
        self.bucket = parent.bucket
        self.prefix = parent.prefix
        self.key = parent.key
        self.key_base = parent.key_base 
        self.dtype = parent.dtype

        # Initialize all size information.
        self.shard_sizes = parent.shard_sizes
        self.parent_slices = []
        self.shape = []
        if isinstance(parent_slices, int):
            parent_slices = [parent_slices]
        for i, parent_slice in enumerate(parent_slices):
            if not isinstance(parent_slice, int):
                # Replace any Nones in slices.
                start = parent_slice.start
                stop = parent_slice.stop
                step = parent_slice.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.parent.shape[i]
                if step is None:
                    step = 1
                self.shape.append((stop - start) // step)
                self.parent_slices.append(slice(start, stop, step))
            else:
                # An integer slice means we are eliminating an axis.
                self.parent_slices.append(parent_slice)
        # Account for the case where slices aren't provided for the trailing indexes.
        self.parent_slices += [slice(0, self.parent.shape[i], 1)
                               for i in range(len(self.parent_slices), len(self.parent.shape))]
    @property
    def blocks_exist(self):
        raise NotImplementedError

    @property
    def blocks_not_exist(self):
        raise NotImplementedError 

    @property
    def blocks(self):
        raise NotImplementedError

    @property
    def block_idxs_exist(self):
        parent_idxs = self.parent.block_idxs_exist() 
        view_idxs =  map(self.__parent_to_view_block_idx__,
                         filter(self.__is_valid_parent_block_idx__, parent_idxs))
        if self.transposed:
            view_idxs = map(reversed, view_idxs)
        return list(view_idxs)

    @property
    def block_idxs_not_exist(self):
        parent_idxs = self.parent.block_idxs_not_exist() 
        view_idxs =  map(self.__parent_to_view_block_idx__,
                         filter(self.__is_valid_parent_block_idx__, parent_idxs))
        if self.transposed:
            view_idxs = map(reversed, view_idxs)
        return list(view_idxs)

    @property
    def block_idxs(self):
        parent_idxs = self.parent.block_idxs() 
        view_idxs = map(self.__parent_to_view_block_idx__,
                        filter(self.__is_valid_parent_block_idx__, parent_idxs))
        if self.transposed:
            view_idxs = map(reversed, view_idxs)
        return list(view_idxs)

    def get_block(self, *block_idx): 
        if self.transposed:
            block_idx = reversed(block_idx)
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        block = self.parent.get_block(*parent_idx)
        if self.transposed:
            block = block.T
        return block

    async def get_block_async(self, loop, *block_idx):
        if self.transposed:
            block_idx = reversed(block_idx)
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        block = self.parent.get_block_async(loop, *parent_idx)
        if self.transposed:
            block = block.T
        return block

    def put_block(self, block, *block_idx):
        if self.transposed:
            block_idx = reversed(block_idx)
            block = block.T
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        return self.parent.put_block(block, *parent_idx)

    async def put_block_async(self, block, loop=None, *block_idx):
        if self.transposed:
            block_idx = reversed(block_idx)
            block = block.T
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        return self.parent.put_block_async(block, loop, *parent_idx)

    def delete_block(self, *block_idx):
        if self.transposed:
            block_idx = reversed(block_idx)
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        return self.parent.delete_block(parent_idx)

    async def delete_block_async(self, loop, *block_idx):
        if self.transposed:
            block_idx = reversed(block_idx)
        parent_idx = self.__view_to_parent_block_idx__(*block_idx)
        return self.parent.delete_block(loop, parent_idx)

    def _block_idxs(self, axis=None):
        parent_axis = self.__view_to_parent_axis__(axis)
        parent_idxs = self.parent._block_idxs(axis=parent_axis) 
        valid_parent_idxs = filter(lambda x: self.__is_valid_parent_block_idx__(x, axis=parent_axis),
                                   parent_idxs)
        view_idxs = map(lambda x: self.__parent_to_view_block_idx__(x, axis=parent_axis),
                        valid_parent_idxs)
        if self.transposed and axis is None:
            view_idxs = map(reversed, idxs)
        return list(view_idxs) 

    def _blocks(self, axis=None):
        raise NotImplementedError

    def __view_to_parent_axis__(self, view_axis):
        if view_axis is None:
            # We make the assumption that subsequent functions can handle None correctly.
            return None
        if self.transposed:
            view_axis = len(self.shape) - view_axis - 1
        num_slices = 0
        for i, parent_slice in enumerate(self.parent_slices):
            if isinstance(parent_slice, slice):
                num_slices += 1
            # Check if we've reached the parent index referred by the view index.
            if num_slices - 1 == view_axis:
                break
        return i 

    def __view_to_parent_block_idx__(self, *view_idx):
        parent_idx = []
        i = 0
        for parent_slice in self.parent_slices:
            if isinstance(parent_slice, int):
                # The view won't have an index for integer slices.
                parent_idx.append(parent_slice)
            else:
                parent_elt = view_idx[i] * parent_slice.step + parent_slice.start
                if parent_elt < 0:
                    raise NotImplementedError
                if parent_elt >= parent_slice.stop:
                    raise IndexError("Array index out of bounds.")
                parent_idx.append(parent_elt)
                i += 1
        return parent_idx

    def __parent_to_view_block_idx__(self, *parent_idx, axis=None):
        print("in", parent_idx, axis)
        # Assign what indices we need to convert.
        if axis is not None:
            parent_slices = [self.parent_slices[axis]]
        else:
            parent_slices = self.parent_slices

        assert len(parent_idx) == len(parent_slices)
        assert self.__is_valid_parent_block_idx__(*parent_idx, axis)
        view_idx = []
        for parent_elt, parent_slice in zip(parent_idx, parent_slices):
            if isinstance(parent_slice, int):
                continue
            view_idx.append((parent_elt - parent_slice.start) // parent_slice.step)
        if axis is not None:
            view_idx = view_idx[0]
        return view_idx

    def __is_valid_parent_block_idx__(self, *parent_idx, axis=None):
        # Assign what indices we need to check.
        if axis is not None:
            parent_slices = [self.parent_slices[axis]]
        else:
            parent_slices = self.parent_slices

        # Now check all relevant indices.
        for parent_elt, parent_slice in zip(parent_idx, parent_slices):
            if parent_elt < 0:
                raise NotImplementedError("Negative indexing not yet supported.")
            if isinstance(parent_slice, int):
                if parent_elt != parent_slice:
                    return False
            else:    
                if parent_elt < parent_slice.start:
                    return False
                if parent_elt >= parent_slice.stop:
                    return False
                if (parent_elt - parent_slice.start) % parent_slice.step != 0:
                    return False
        return True  

    def __str__(self):
        slice_reps = [] 
        last_slice = 0 
        for parent_slice, parent_shape in zip(self.parent_slices, self.parent.shape):
            if parent_slice != slice(0, parent_shape, 1):
                last_slice += 1
            if isinstance(parent_slice, int):
                slice_reps.append(str(parent_slice))
            else:
                if parent_slice.step == 1:
                    step_rep = ""
                else:
                    step_rep = ":" + parent_slice.step 
                if parent_slice.start == 0:
                    start_rep = ""
                else:
                    start_rep = str(parent_slice.start)
                if stop_rep == self.parent.shape[i]:
                    stop_rep = ""
                else:
                    stop_rep = str(parent_slice.stop)
                slice_reps.append(start_rep + ":" + stop_rep + step_rep)
        rep = self.parent.__str__() 
        if last_slice != 0:
            rep += "[" + ",".join(slice_reps[:last_slice]) + "]"
        if self.transposed:
            rep += ".T"
        return rep

class Scalar(BigMatrix):
    def __init__(self, key,
                 bucket=DEFAULT_BUCKET,
                 prefix='numpywren.objects/',
                 parent_fn=None, 
                 dtype='float64'):
        self.bucket = bucket
        self.prefix = prefix
        self.key = key
        self.key_base = prefix + self.key + "/"
        self.dtype = dtype
        self.transposed = False
        self.parent_fn = parent_fn
        self.shard_sizes = [1]
        self.shape = [1]

    def numpy(self, workers=1):
        return BigMatrix.get_block(self, 0)[0]

    def get(self, workers=1):
        return BigMatrix.get_block(self, 0)[0]

    def put(self, value):
        value = np.array([value])
        BigMatrix.put_block(self, value, 0)

    def __str__(self):
        rep = "Scalar({0})".format(self.key)
        return rep




class BigSymmetricMatrix(BigMatrix):

    def __init__(self, key,
                 shape=None,
                 shard_sizes=[],
                 bucket=DEFAULT_BUCKET,
                 prefix='numpywren.objects/',
                 dtype=np.float64,
                 parent_fn=None,
                 write_header=False,
                 lambdav = 0.0):
        BigMatrix.__init__(self, key=key, shape=shape, shard_sizes=shard_sizes, bucket=bucket, prefix=prefix, dtype=dtype, parent_fn=parent_fn, write_header=write_header)
        self.symmetric = True
        self.lambdav = lambdav


    @property
    def T(self):
        return self


    def _symmetrize_idx(self, block_idx):
        if np.all(block_idx[0] > block_idx[-1]):
            return tuple(block_idx)
        else:
            return tuple(reversed(block_idx))

    def _symmetrize_all_idxs(self, all_block_idxs):
        return sorted(list(set((map(lambda x: tuple(self._symmetrize_idx(x)), all_block_idxs)))))

    def _blocks(self, axis=None):
        if axis is None:
            block_idxs = self._block_idxs()
            blocks = [self.__block_idx_to_real_idx__(x) for x in block_idxs]
            return blocks
        elif (type(axis) != int):
            raise Exception("Axis must be integer")
        else:
            return super()._blocks(axis=axis)

    def _block_idxs(self, axis=None):
        all_block_idxs = super()._block_idxs(axis=axis)
        if (axis == None):
            valid_block_idxs = self._symmetrize_all_idxs(all_block_idxs)
            return valid_block_idxs
        else:
            return all_block_idxs

    async def get_block_async(self, loop=None, *block_idx):
        if (loop == None):
            loop = asyncio.get_event_loop()
        # For symmetric matrices it suffices to only read from lower triangular
        flipped = False
        block_idx_sym = self._symmetrize_idx(block_idx)
        if block_idx_sym != block_idx:
            flipped = True
        key = self.__shard_idx_to_key__(block_idx_sym)
        exists = await key_exists_async(self.bucket, key)
        if (not exists and self.parent_fn == None):
            raise Exception("Key does not exist, and no parent function prescripted")
        elif (not exists and self.parent_fn != None):
            X_block = self.parent_fn(self, *block_idx_sym)
        else:
            bio = await self.__s3_key_to_byte_io__(key, loop=loop)
            X_block = np.load(bio).astype(self.dtype, loop)
        if (flipped):
            X_block = X_block.T
        if (len(list(set(block_idx))) == 1):
            idxs = np.diag_indices(X_block.shape[0])
            X_block[idxs] += self.lambdav
        return X_block

    async def put_block_async(self, block, loop=None, *block_idx):
        if (loop == None):
            loop = asyncio.get_event_loop()
        block_idx_sym = self._symmetrize_idx(block_idx)
        if block_idx_sym != block_idx:
            flipped = True
            block = block.T
        real_idxs = self.__block_idx_to_real_idx__(block_idx_sym)
        current_shape = tuple([e - s for s,e in real_idxs])
        if (block.shape != current_shape):
            raise Exception("Incompatible block size: {0} vs {1}".format(block.shape, current_shape))
        key = self.__shard_idx_to_key__(block_idx)
        block = block.astype(self.dtype)
        return await self.__save_matrix_to_s3__(block, key, loop)


    async def delete_block_async(self, loop, *block_idx):
        if (loop == None):
            loop = asyncio.get_event_loop()
        block_idx_sym = self._symmetrize_idx(block_idx)
        if block_idx_sym != block_idx:
            flipped = True
        key = self.__shard_idx_to_key__(block_idx_sym)
        aiobotocore.get_session(loop=loop)
        async with session.create_client('s3', use_ssl=False, verify=False, region_name="us-west-2") as client:
            resp = client.delete_object(Key=key, Bucket=self.bucket)
        return resp

