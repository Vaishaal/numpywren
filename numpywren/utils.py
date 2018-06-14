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

