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

def merge_dictionaries(*args):
    res = []
    keys = []
    for arg in args:
        res += list(arg.items())
        keys += list(arg.keys())
    if (len(keys) != len(set(keys))):
        raise Exception("can only merge dictionaries with unique keys")
    return dict(res)
