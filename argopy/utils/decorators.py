from functools import wraps
import warnings
import logging

import xarray as xr
from decorator import decorator
from typing import List

from ..errors import NoDataLeft


log = logging.getLogger("argopy.utils.decorators")


class DocInherit(object):
    """Docstring inheriting method descriptor

    The class itself is also used as a decorator

    Usage:

    class Foo(object):
        def foo(self):
            "Frobber"
            pass

    class Bar(Foo):
        @doc_inherit
        def foo(self):
            pass

    Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"

    src: https://code.activestate.com/recipes/576862/
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):
        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=("__name__", "__module__"))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=("__name__", "__module__"))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func


doc_inherit = DocInherit


def deprecated(reason: str = None, version: str = None, ignore_caller: List = []):
    """Deprecation warning decorator

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    reason: str, optional, default=None
        Text message to send with deprecation warning
    version: str, optional, default=None
    ignore_caller: List, optional, default=[]

    Examples
    --------
    The @deprecated can be used with a 'reason' and a 'version'

        .. code-block:: python

           @deprecated("please, use another function", version='0.2')
           def old_function(x, y):
             pass

    or without:

        .. code-block:: python

           @deprecated
           def old_function(x, y):
             pass

    The @deprecated can also be ignored from specific callers.

        .. code-block:: python

           @deprecated("please, use another function", version='0.2', ignore_caller='postprocessing')
           def old_function(x, y):
             pass

    References
    ----------
    This decorator is largely inspired by https://stackoverflow.com/a/40301488
    """
    import inspect
    ignore_caller = [ignore_caller]

    if isinstance(reason, str):

        def decorator(func):
            if inspect.isclass(func):
                fmt = "\nCall to deprecated class '{name}' ({reason})"
            else:
                fmt = "\nCall to deprecated function '{name}' ({reason})"
            if version is not None:
                fmt = "%s -- Deprecated since version {version}" % fmt

            @wraps(func)
            def new_func(*args, **kwargs):
                raise_deprec = True
                stack = inspect.stack()
                for s in stack:
                    if "<module>" in s.function:
                        break
                    elif s.function in ignore_caller:
                        raise_deprec = False

                if raise_deprec:
                    warnings.simplefilter("always", DeprecationWarning)
                    warnings.warn(
                        fmt.format(name=func.__qualname__, reason=reason, version=version),
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
                    warnings.simplefilter("default", DeprecationWarning)
                else:
                    log.warning(fmt.format(name=func.__qualname__, reason=reason, version=version))

                return func(*args, **kwargs)

            return new_func

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason

        if inspect.isclass(func):
            fmt = "\nCall to deprecated class '{name}'."
        else:
            fmt = "\nCall to deprecated function '{name}'."

        @wraps(func)
        def new_func(*args, **kwargs):
            raise_deprec = True
            stack = inspect.stack()
            for s in stack:
                if "<module>" in s.function:
                    break
                elif s.function in ignore_caller:
                    raise_deprec = False

            if raise_deprec:
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt.format(name=func.__qualname__),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
            else:
                log.warning(fmt.format(name=func.__qualname__, reason=reason))

            return func(*args, **kwargs)

        return new_func

    else:
        raise TypeError(repr(type(reason)))


@decorator
def raiseNoDataLeft(func, msg='No Data left after post-processing', *args, **kwargs):
    outputs = func(*args, **kwargs)
    if isinstance(outputs, xr.Dataset) and 'N_POINTS' in outputs and len(outputs['N_POINTS']) == 0:
        raise NoDataLeft(msg)
    else:
        return outputs
