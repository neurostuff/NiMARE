def njit(*args, **kwargs):
    """Mock njit decorator."""
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

def jit(*args, **kwargs):
    return njit(*args, **kwargs)

def prange(*args, **kwargs):
    return range(*args, **kwargs)

def literal_unroll(val):
    return val