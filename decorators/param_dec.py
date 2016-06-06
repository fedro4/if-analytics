import inspect
from functools import wraps

def dictparams(func):
    aspe = None
    if func.func_closure is not None:
        aspe = inspect.getargspec(func.func_closure[0].cell_contents) 
    else:
        aspe = inspect.getargspec(func)  
    argnames = aspe.args
    needed_argnames = argnames
    if aspe.defaults is not None:
        needed_argnames = needed_argnames[:-len(aspe.defaults)]
    

    @wraps(func)
    def wrapper(*args, **kwargs):
        d = {}
        if len(args) > 0:
            if not isinstance(args[0], dict):
                raise TypeError("the only non-keyword arg accepted by %s is (at most) one dict (due to @dictparams decorator)" % func.func_name)
            d = dict(args[0])
        if len(args) > 1:
            raise TypeError("%s can only be used with one dictionary arg and optionally keyword args (due to @dictparams decorator)" % func.func_name)
        d.update(kwargs)
        
        for k in d.keys():
            if not k in argnames:
                if k in kwargs.keys(): # this was explicitly passed, but is not an argument the function takes
                    raise TypeError("%s got an unexpected keyword argument '%s'" % (func.func_name, k))
                del d[k]
        
        # find out which arguments are missing
        missing = []
        for a in needed_argnames:
            if not a in d.keys():
                missing.append(a)
        if len(missing) > 0:
            raise TypeError("%s is missing arguments: %s" % (func.func_name, ",".join(["'"+m+"'" for m in missing])))
        return func(**d)

    if wrapper.__doc__ is not None:
        # this is a rather sad hack cause the function signature of a decorated function is not preserved for the help
        doclines = wrapper.__doc__.split('\n')
        newdocline = "\nThis function takes the keyword arguments: %s\nA dict containing default parameter values may be passed as the only non-keyword argument." % ", ".join(argnames)
        wrapper.__doc__ = '\n'.join([doclines[0], newdocline] + doclines[1:])
    return wrapper



