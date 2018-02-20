import ctypes
import numpy as np
import load_shared_lib
import concurrent.futures as fs
import time
import multiprocessing as mp
import argparse


CPU_COUNT = mp.cpu_count()

class FastIO(object):
    ''' A fairly low level api to read/write from S3 using the python ctypes API '''

    def __init__(self, so_bucket="numpywrenpublic"):
        self.so_bucket = so_bucket
        self.so_key = "fastio"
        self.__so_cached = None
        self.__api_started = False

    @property
    def so(self):
        if (self.__so_cached) == None:
            return load_shared_lib.load_shared_lib(self.so_key, self.so_bucket)
        else:
            return self.__so_cached

    def start_api(self):
        ''' Must be called before any of the other functions '''
        if (not self.__api_started):
            self.so.start_api()
            self.__api_started = True
        else:
            raise Exception("Cannot start API more than once")

    def stop_api(self):
        ''' Must be called before any of the other functions '''
        if (self.__api_started):
            self.so.stop_api()
            self.__api_started = False
        else:
            raise Exception("Cannot stop API before start")




    def get_object(self, ptr, nbytes, bucket, key):
        ''' Download an object nbytes long from s3://bucket/key and store it in ptr'''
        assert(self.__api_started)
        so = self.so
        so.get_object.arg_types = [ctypes.c_void_p, ctypes.c_long, ctypes.c_char_p, ctypes.c_char_p];
        return so.get_object(ptr, nbytes, bucket.encode(), key.encode())

    def get_objects(self, ptrs, buffer_sizes, buckets, keys, threads=CPU_COUNT):
        ''' Upload a list of objects from ptrs to s3://bucket
        '''
        assert(self.__api_started)
        assert(len(ptrs) == len(buffer_sizes) == len(buckets) == len(keys))

        num_objects = len(ptrs)

        void_star_star  = ctypes.c_void_p * num_objects
        char_star_star = ctypes.c_char_p * num_objects
        long_star = ctypes.c_long * num_objects

        c_buffers_array = void_star_star(*ptrs)
        c_keys_array = char_star_star(*keys)
        c_buckets_array = char_star_star(*buckets)
        c_buffer_sizes_array = long_star(*buffer_sizes)
        self.so.put_objects.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_long, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        self.so.get_objects(c_buffers_array, num_objects, c_buffer_sizes_array, c_buckets_array, c_keys_array, threads)


    def put_objects(self, ptrs, buffer_sizes, buckets, keys, threads=CPU_COUNT):
        ''' Upload a list of objects from ptrs to s3://bucket
        '''
        assert(self.__api_started)
        assert(len(ptrs) == len(buffer_sizes) == len(buckets) == len(keys))

        num_objects = len(ptrs)

        void_star_star  = ctypes.c_void_p * num_objects
        char_star_star = ctypes.c_char_p * num_objects
        long_star = ctypes.c_long * num_objects

        c_buffers_array = void_star_star(*ptrs)
        c_keys_array = char_star_star(*keys)
        c_buckets_array = char_star_star(*buckets)
        c_buffer_sizes_array = long_star(*buffer_sizes)
        self.so.put_objects.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_long, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        self.so.put_objects(c_buffers_array, num_objects, c_buffer_sizes_array, c_buckets_array, c_keys_array, threads)

    def put_object(self, ptr, nbytes, bucket, key):
        ''' Upload an object nbytes long from ptr to s3://bucket/key
        '''
        assert(self.__api_started)
        so = self.sio
        so.put_object.arg_types = [ctypes.c_void_p, ctypes.c_long, ctypes.c_char_p, ctypes.c_char_p];
        return so.put_object(ptr, nbytes, bucket.encode(), key.encode())


    def cache_so(self):
        ''' Cache the fastio.so in memory
          NOTE: THIS WILL MAKE THIS OBJECT UNSERIALIZABLE PLEASE CALL
          .uncache_so(self) if you want to send this object over the wire
         '''
        self.__so_cached = self.so

    def uncache_so(self):
        ''' Uncache the fastio.so from memory so object is serializable
         '''
        self.__so_cached = None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastIO test')
    parser.add_argument('obj_size', type=int)
    parser.add_argument('num_objects', type=int)
    parser.add_argument('bucket', type=str)
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()
    fio = FastIO()
    fio.cache_so()
    fio.start_api()
    obj_size = args.obj_size
    num_objects = args.num_objects
    prefix = args.prefix
    mat_in = np.ones(obj_size*num_objects, np.uint8)
    mat_out = np.zeros(obj_size*num_objects, np.uint8)
    print(mat_out)
    print(mat_in)
    executor = fs.ThreadPoolExecutor(1)
    futures = []
    start = time.time()
    ptrs_in = []
    ptrs_out = []
    buckets = []
    keys = []
    buffer_sizes = []
    start = time.time()
    for i in range(num_objects):
        mat_ptr_in = mat_in[i*obj_size:(i+1)*obj_size]
        mat_ptr_out =  mat_out[i*obj_size:(i+1)*obj_size]
        ptr_in = ctypes.c_void_p(mat_ptr_in.ctypes.data)
        ptrs_in.append(ptr_in)
        ptr_out = ctypes.c_void_p(mat_ptr_out.ctypes.data)
        ptrs_out.append(ptr_out)
        key = (prefix + "/" + str(i))
        buckets.append(ctypes.c_char_p("pictureweb".encode()))
        keys.append(ctypes.c_char_p(key.encode()))
        buffer_sizes.append(ctypes.c_long(mat_ptr_in.nbytes))
    fio.put_objects(ptrs_in, buffer_sizes, buckets, keys, threads=num_objects)
    end = time.time()
    print("Bytes written ", mat_in.nbytes)
    print("Write Time",  end - start)
    print("Write GB/s", mat_in.nbytes/((end - start)*1e9))

    start = time.time()
    fio.get_objects(ptrs_out, buffer_sizes, buckets, keys, threads=num_objects)
    end = time.time()
    print("Bytes written ", mat_in.nbytes)
    print("Read Time",  end - start)
    print("READ GB/s", mat_in.nbytes/((end - start)*1e9))
    print(mat_out)
    assert(np.all(mat_out == mat_in))






















