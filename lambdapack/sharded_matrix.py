import concurrent.futures as fs
import io
import itertools
import os
import time

import boto3
import cloudpickle
import numpy as np


def list_all_keys(bucket, prefix):
    client = boto3.client('s3')
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    keys = list(map(lambda x: x['Key'], objects['Contents']))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))

def block_key_to_block(key):
    try:
        block_key = key.split("/")[-1]
        blocks_split = block_key.split("_")
        b0_start = int(blocks_split[0])
        b0_end = int(blocks_split[1])
        b1_start = int(blocks_split[3])
        b1_end = int(blocks_split[4])
        return ((b0_start, b0_end), (b1_start, b1_end))
    except Exception as e:
        return None


class ShardedMatrix(object):
    def __init__(self, key,
                 shape=None,
                 shard_size_0=None,
                 shard_size_1=None,
                 bucket=None,
                 prefix='pywren.linalg/'):

        if bucket is None:
            bucket = os.environ.get('PYWREN_LINALG_BUCKET')
            if bucket is None:
                raise Exception("bucket not provided and environment variable \
                        PYWREN_LINALG_BUCKET not provided")
        self.bucket = bucket
        self.prefix = prefix
        self.key = key
        self.key_base = prefix + self.key + "/"
        header = self.__read_header__()
        self.pwex = None
        if header is None and shape is None:
            raise Exception("header doesn't exist and no shape provided")
        if not (header is None) and shape is None:
            self.shard_size_0 = header.shard_size_0
            self.shard_size_1 = header.shard_size_1
            self.shape = header.shape
        else:
            self.shape = shape
            self.shard_size_0 = shard_size_0
            self.shard_size_1 = shard_size_1

        if self.shard_size_0 is None:
            raise Exception("No shard_0_size provided")
        if self.shard_size_1 is None:
            self.shard_size_1 = self.shape[1]

        self.symmetric = False
        self.shard_sizes = (self.shard_size_0, self.shard_size_1)
        self.__write_header__()

    @property
    def blocks_exist(self):
        #slow
        prefix = self.prefix + self.key
        all_keys = list_all_keys(self.bucket, prefix)
        return list(filter(lambda x: x != None, map(block_key_to_block, all_keys)))

    @property
    def blocks(self):
        return self._blocks()

    @property
    def block_idxs_exist(self):
        all_block_idxs = self.block_idxs
        all_blocks = self.blocks
        blocks_exist = set(self.blocks_exist)
        block_idxs_exist = []
        for i, block in enumerate(all_blocks):
            if block in blocks_exist:
                block_idxs_exist.append(all_block_idxs[i])
        return block_idxs_exist

    @property
    def blocks_not_exist(self):
        blocks = set(self.blocks)
        block_exist = set(self.blocks_exist)
        return list(filter(lambda x: x, list(block_exist.symmetric_difference(blocks))))

    @property
    def block_idxs_not_exist(self):
        block_idxs = set(self.block_idxs)
        block_idxs_exist = set(self.block_idxs_exist)
        return list(filter(lambda x: x, list(block_idxs_exist.symmetric_difference(block_idxs))))




    @property
    def block_idxs(self):
        return self._block_idxs()

    def _blocks(self, axis=None):

        blocks_x = [(i, i + self.shard_size_0) for i in range(0, self.shape[0], self.shard_size_0)]

        if blocks_x[-1][1] > self.shape[0]:
            blocks_x.pop()

        if blocks_x[-1][1] < self.shape[0]:
            blocks_x.append((blocks_x[-1][1], self.shape[0]))


        blocks_y = [(i, i + self.shard_size_1) for i in range(0, self.shape[1], self.shard_size_1)]

        if blocks_y[-1][1] > self.shape[1]:
            blocks_y.pop()

        if blocks_y[-1][1] < self.shape[1]:
            blocks_y.append((blocks_y[-1][1], self.shape[1]))

        if axis is None:
            return list(itertools.product(blocks_x, blocks_y))
        elif axis == 0:
            return blocks_x
        elif axis == 1:
            return blocks_y
        else:
            raise Exception("Invalid Axis")

    def _block_idxs(self, axis=None):
        blocks_x = list(range(len(self._blocks(axis=0))))
        blocks_y = list(range(len(self._blocks(axis=1))))

        if axis is None:
            return list(itertools.product(blocks_x, blocks_y))
        elif axis == 0:
            return blocks_x
        elif axis == 1:
            return blocks_y
        else:
            raise Exception("Invalid Axis")

    def idx_to_block_idx(self, idx_1, idx_2):
        blocks_x = self._blocks(0)
        blocks_y = self._blocks(1)

        block_x = -1
        block_y = -1

        for i, (blk_start, blk_end) in enumerate(blocks_x):
            if blk_start <= idx_1 and blk_end > idx_1:
                block_x = i
                offset_x = idx_1 - blk_start

        for i, (blk_start, blk_end) in enumerate(blocks_y):
            if blk_start <= idx_2 and blk_end > idx_2:
                block_y = i
                offset_y = idx_2 - blk_start

        if block_x == -1:
            raise Exception("Index 0 out of bounds")

        if block_y == -1:
            raise Exception("Index 1 out of bounds")

        return block_x, block_y, offset_x, offset_y


    def __getitem__(self, idxs):
        idx_1, idx_2 = idxs

        if isinstance(idx_1, slice):
            raise Exception("Slicing in first index not implemented")

        if isinstance(idx_2, slice):
            if idx_2.start != None or idx_2.step != None or idx_2.stop != None:
                raise Exception("Only full row slices supported")
            blocks_y_idxs = self._block_idxs(axis=1)
            blocks = []
            block_x, block_y, offset_x, offset_y = self.idx_to_block_idx(idx_1, 0)
            for blk_idx in blocks_y_idxs:
                blocks.append(self.get_block(block_x, blk_idx))
            return np.hstack(blocks)[offset_x, :]

        else:
            block_x, block_y, offset_x, offset_y = self.idx_to_block_idx(idx_1, idx_2)
            block_data = self.get_block(block_x, block_y)
            return block_data[offset_x, offset_y]

    def __get_matrix_shard_key__(self, start_0, end_0, start_1, end_1, replicate=0):
            rep = str(replicate)
            key_string = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(start_0, end_0, self.shard_size_0, start_1, end_1, self.shard_size_1, rep)
            return self.key_base + key_string



    def __read_header__(self):
        client = boto3.client('s3')
        try:
            key = self.key_base + "header"
            header = cloudpickle.loads(client.get_object(Bucket=self.bucket, Key=key)['Body'].read())
        except:
            header = None
        return header


    def __write_header__(self):
        client = boto3.client('s3')
        key = self.key_base + "header"
        client.put_object(Key=key, Bucket = self.bucket, Body=cloudpickle.dumps(self), ACL="bucket-owner-full-control")
        return 0

    def __shard_idx_to_key__(self, shard_0, shard_1, replicate=0):
        N = self.shape[0]
        D = self.shape[1]
        start_0 = shard_0*self.shard_size_0
        start_1 = shard_1*self.shard_size_1
        end_0 = min(start_0+self.shard_size_0, N)
        end_1 = min(start_1+self.shard_size_1, D)
        key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, replicate)
        return key

    def __s3_key_to_byte_io__(self, key):
        n_tries = 0
        max_n_tries = 5
        bio = None
        client = boto3.client('s3')
        while bio is None and n_tries <= max_n_tries:
            try:
                bio = io.BytesIO(client.get_object(Bucket=self.bucket, Key=key)['Body'].read())
            except Exception as e:
                n_tries += 1
        if bio is None:
            raise Exception("S3 Read Failed")
        return bio

    def __save_matrix_to_s3__(self, X, out_key, client=None):
        if (client == None):
            client = boto3.client('s3')
        outb = io.BytesIO()
        np.save(outb, X)
        response = client.put_object(Key=out_key, Bucket=self.bucket, Body=outb.getvalue(),ACL="bucket-owner-full-control")
        return response

    def shard_matrix(self, X, executor=None, n_jobs=1):
        print("Sharding matrix..... of shape {0}".format(X.shape))
        bidxs = self.block_idxs
        blocks = self.blocks
        if (executor is None):
            executor = fs.ThreadPoolExecutor(n_jobs)

        futures = []
        for ((bidx_0, bidx_1),(block_0, block_1)) in zip(bidxs, blocks):
            bstart_0, bend_0 = block_0
            bstart_1, bend_1 = block_1
            future = executor.submit(self.put_block, bidx_0, bidx_1, X[bstart_0:bend_0, bstart_1:bend_1])
            futures.append(future)
        fs.wait(futures)
        return 0


    def get_blocks(self, blocks_0, block_1):
        client = boto3.client('s3')
        blocks = [self.get_block(block_0, block_1, client) for block_0 in blocks_0]
        return np.vstack(blocks)

    def get_blocks_mmap(self, blocks_0, blocks_1, mmap_loc, mmap_shape, dtype='float32', row_offset=0, col_offset=0):
        X_full = np.memmap(mmap_loc, dtype=dtype, mode='r+', shape=mmap_shape)

        b_start = col_offset*self.shard_size_1
        for i,block_1 in enumerate(blocks_1):
            width = min(self.shard_size_1, self.shape[1] - (block_1)*self.shard_size_1)
            b_end = b_start + width
            for i,block_0 in enumerate(blocks_0):
                block = self.get_block(block_0, block_1)
                curr_row_block = (row_offset+i)
                sidx = curr_row_block*self.shard_size_0
                eidx = min((curr_row_block+1)*self.shard_size_0, self.shape[0])
                try:
                    X_full[sidx:eidx, b_start:b_end] = block
                except Exception as e:
                    print("SIDX", sidx)
                    print("EIDX  ", eidx)
                    print("BEND", b_end)
                    print("BSTART", b_start)
                    print(width, block_1, block_0, block.shape, (eidx-sidx, b_end-b_start))
                    print("X_FULL PART SHAPE ", X_full[sidx:eidx, b_start:b_end].shape)
                    print("X FULL SHAPE ", X_full.shape)
                    print(e)
                    raise
            b_start = b_end
        X_full.flush()
        return (mmap_loc, mmap_shape, dtype)


    def get_block(self, block_0, block_1, client=None, flip=False):
        try:
            if (client == None):
                client = boto3.client('s3')
            else:
                client = client

            if (flip):
                block_0, block_1 = block_1, block_0

            s = time.time()
            if ((block_0, block_1) in self.__cached_blocks):
                X_block = self.__cached_blocks[(block_0, block_1)]
            else:
                r = np.random.choice(self.replication_factor, 1)[0]
                key = self.__shard_idx_to_key__(block_0, block_1, r)
                bio = self.__s3_key_to_byte_io__(key)
                e = time.time()
                s = time.time()
                X_block = np.load(bio)
                e = time.time()

            if (flip):
                X_block = X_block.T
        except:
            print(block_0, block_1)
            raise
        return X_block

    def put_block(self, block_0, block_1, block):

        start_0 = block_0*self.shard_size_0
        end_0 = min(block_0*self.shard_size_0 + self.shard_size_0, self.shape[0])
        shape_0 = end_0 - start_0

        start_1 = block_1*self.shard_size_1
        end_1 = min(block_1*self.shard_size_1 + self.shard_size_1, self.shape[1])
        shape_1 = end_1 - start_1


        if (block.shape != (shape_0, shape_1)):
            raise Exception("Incompatible block size: {0} vs {1}".format(block.shape, (shape_0,shape_1)))

        for i in range(self.replication_factor):
            key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, i)
            self.__save_matrix_to_s3__(block, key)


    def delete_block(self, block_0, block_1):
        client = boto3.client('s3')
        if ((block_0, block_1) in self.__cached_blocks):
            del self.__cached_blocks[(block_0, block_1)]

        start_0 = block_0*self.shard_size_0
        end_0 = min(block_0*self.shard_size_0 + self.shard_size_0, self.shape[0])
        shape_0 = end_0 - start_0

        start_1 = block_1*self.shard_size_1
        end_1 = min(block_1*self.shard_size_1 + self.shard_size_1, self.shape[1])
        shape_1 = end_1 - start_1

        deletions = []
        for i in range(self.replication_factor):
            key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, i)
            deletions.append(client.delete_object(Key=key, Bucket=self.bucket))
        return deletions

    def dumps(self):
        return cloudpickle.dumps(self)


class ShardedSymmetricMatrix(ShardedMatrix):
    def __init__(self, key, data=None, shape=None, shard_size_0=4096, shard_size_1=None, bucket=None, prefix='pywren.linalg/', transposed = False, diag_offset=0.0):
        ShardedMatrix.__init__(self, key, data, shape, shard_size_0, shard_size_1, bucket, prefix, transposed)
        self.symmetric = True
        self.diag_offset = diag_offset
        self.__cached_blocks = {}

    def _blocks(self, axis=None):

        blocks_x = [(i, i + self.shard_size_0) for i in range(0, self.shape[0], self.shard_size_0)]

        if (blocks_x[-1][1] > self.shape[0]):
            blocks_x.pop()

        if (blocks_x[-1][1] < self.shape[0]):
            blocks_x.append((blocks_x[-1][1], self.shape[0]))


        blocks_y = [(i, i + self.shard_size_1) for i in range(0, self.shape[1], self.shard_size_1)]

        if (blocks_y[-1][1] > self.shape[1]):
            blocks_y.pop()

        if (blocks_y[-1][1] < self.shape[1]):
            blocks_y.append((blocks_y[-1][1], self.shape[1]))

        all_pairs = list(itertools.product(blocks_x, blocks_y))
        sorted_pairs = map(lambda x: tuple(sorted(x)), all_pairs)

        valid_blocks = sorted(list(set(sorted_pairs)))

        if (axis==None):
            return valid_blocks
        elif (axis == 0):
            return blocks_x
        elif (axis == 1):
            return blocks_y
        else:
            raise Exception("Invalid Axis")

    def _block_idxs(self, axis=None):
        all_block_idxs = ShardedMatrix._block_idxs(self, axis=axis)
        if (axis == None):
            sorted_pairs = map(lambda x: tuple(sorted(x)), all_block_idxs)
            valid_blocks = sorted(list(set(sorted_pairs)))
            return valid_blocks
        else:
            return all_block_idxs



    def get_block(self, block_0, block_1):
        # For symmetric matrices it suffices to only read from lower triangular
        try:
            flipped = False
            if block_1 > block_0:
                flipped = True
                block_0, block_1 = block_1, block_0

            key = self.__shard_idx_to_key__(block_0, block_1)
            bio = self.__s3_key_to_byte_io__(key)
            X_block = np.load(bio)

            if block_0 == block_1 and X_block.shape[0] == X_block.shape[1]:
                diag = np.diag_indices(X_block.shape[0])
                X_block[diag] += self.diag_offset
        except:
            print(block_0, block_1)
            raise

        return X_block

    def put_block(self, block_0, block_1, block):

        if block_1 > block_0:
            block_0, block_1 = block_1, block_0
            block = block.T

        start_0 = block_0*self.shard_size_0
        end_0 = min(block_0*self.shard_size_0 + self.shard_size_0, self.shape[0])
        shape_0 = end_0 - start_0

        start_1 = block_1*self.shard_size_1
        end_1 = min(block_1*self.shard_size_1 + self.shard_size_1, self.shape[1])
        shape_1 = end_1 - start_1


        if block.shape != (shape_0, shape_1):
            raise Exception("Incompatible block size: {0} vs {1}"
                            .format(block.shape, (shape_0,shape_1)))

        key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1)
        self.__save_matrix_to_s3__(block, key)
        return self.__save_matrix_to_s3__(block, key)

    @property
    def blocks_exist(self):
        #slow
        prefix = self.prefix + self.key
        all_keys = list_all_keys(self.bucket, prefix)
        blocks_exist = list(filter(lambda x: x != None, map(block_key_to_block, all_keys)))
        sorted_blocks_exist = map(lambda x: tuple(sorted(x)), blocks_exist)
        valid_blocks_exist = set(sorted(list(set(sorted_blocks_exist))))
        return valid_blocks_exist


    @property
    def blocks_not_exist(self):
        blocks = set(self.blocks)
        blocks_exist = self.blocks_exist
        sorted_blocks_exist = map(lambda x: tuple(sorted(x)), blocks_exist)
        valid_blocks_exist  = set(sorted(list(set(sorted_blocks_exist))))

        return list(filter(lambda x: x, list(blocks.difference(valid_blocks_exist))))

    @property
    def block_idxs_not_exist(self):
        block_idxs = set(self.block_idxs)
        block_idxs_exist = set(self.block_idxs_exist)
        sorted_block_idxs_exist = list(map(lambda x: tuple(sorted(x)), block_idxs_exist))
        valid_block_idxs_exist = set(sorted(list(set(sorted_block_idxs_exist))))
        return list(filter(lambda x: x, list(block_idxs.difference(valid_block_idxs_exist))))


    def delete_block(self, block_0, block_1):
        deletions = []
        client = boto3.client('s3')
        if block_1 > block_0:
            block_0, block_1 = block_1, block_0

        start_0 = block_0*self.shard_size_0
        end_0 = min(block_0*self.shard_size_0 + self.shard_size_0, self.shape[0])

        start_1 = block_1*self.shard_size_1
        end_1 = min(block_1*self.shard_size_1 + self.shard_size_1, self.shape[1])

        key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1)
        deletions.append(client.delete_object(Key=key, Bucket=self.bucket))
        return deletions
