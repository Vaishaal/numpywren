import boto3
import time

BACKOFF = 1
MAX_TRIES = 100

def convert_to_slice(l):
    if l is None:
        return slice(None, None, None)
    elif isinstance(l, int):
        return slice(l, l + 1, 1)
    elif not isinstance(l, slice):
        start = None
        stop = None
        step = None
        if len(l) == 1:
            stop = l[0]
        elif len(l) == 2:
            start = l[0]
            stop = l[1]
        elif len(l) == 3:
            start = l[0]
            stop = l[1]
            step = l[2]
        else:
            raise ValueError("Expected slices of length 1 to 3.")
        return slice(start, stop, step)
    else:
        raise ValueError("Could not convert to slice.")

def remove_duplicates(l):
    try: 
        return list(set(l))
    except TypeError:
        pass

    try:
        [dict(t) for t in set([tuple(d.items()) for d in l])]
    except AttributeError:
        pass

    new_list = []
    for elt in l:
        if elt not in new_list:
            new_list.append(elt)
    return new_list 

def merge_dicts(*args):
    res = []
    keys = []
    for arg in args:
        res += list(arg.items())
        keys += list(arg.keys())
    if (len(keys) != len(set(keys))):
        raise Exception("can only merge dictionaries with unique keys")
    return dict(res)

def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if n == 0: return []
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_object_with_backoff(s3_client, bucket, key, max_tries=MAX_TRIES, backoff=BACKOFF, **extra_get_args):
    num_tries = 0
    while (num_tries < max_tries):
        try:
            obj_bytes = s3_client.get_object(Bucket=bucket, Key=key, **extra_get_args)["Body"].read()
            break
        except:
            time.sleep(backoff)
            backoff *= 2
            num_tries += 1
            obj_bytes = None
    if (obj_bytes is None):
        raise Exception("S3 Download Failed")
    return obj_bytes 
