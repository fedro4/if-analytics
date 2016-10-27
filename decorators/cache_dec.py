""" a simple caching of function return values
using the decorator "cached", e.g.

    @cached
    def foo(a, b, c):
        return a*b-c

will cache the result of the calculation foo does, which of course better not be this trivial.
works also for numpy arrays in the parameters.
should of course only be used on functions that do not depend on global parameters (as their state would not be cashed)
"""

import hashlib
import numpy as np
from functools import wraps

cache = {}
hits = 0
misses = 0
no_caching = False

def cached(func):
    global cache
    def hashit(a):
        # builtin hash does weird things with complex number with integer real (or imag?) part : hash(1.5j-1) == hash(1.5j-2)
        return (a.__hash__() if not isinstance(a,np.ndarray) else hashlib.sha1(a).hexdigest())

    @wraps(func)
    def wrapper(*args, **kwargs): # kwargs does not work yet!
        global misses, hits
        key = tuple([func.__name__]) + tuple(("",hashit(a)) for a in args) + tuple((k,hashit(v)) for k, v in sorted(kwargs.items()))
        if no_caching:
            return func(*args, **kwargs)
        elif not cache.has_key(key):
            #print func.__name__ + " missed " + str(key)
            cache[key] = func(*args, **kwargs)
            misses += 1
        else:
            hits += 1
            #print func.__name__ + " hit"
        return cache[key]
    return wrapper
       
def clear_cache():
    global cache, misses, hits
    cache = {}
    hits = 0
    misses = 0
