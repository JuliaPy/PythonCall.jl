"""Julia value wrappers from src/JlWrap/callback.jl."""

from juliacall import ValueBase

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class CallbackValue(ValueBase):
    __module__ = "juliacall"
    __slots__ = ()
    def __repr__(self):
        if self._jl_isnull():
            return "<jl NULL>"
        else:
            return self._jl_callmethod(_jl_methods["__repr__"])
    def __str__(self):
        if self._jl_isnull():
            return "NULL"
        else:
            return self._jl_callmethod(_jl_methods["__str__"])
    def __call__(self, *args, **kwargs):
        return self._jl_callmethod(_jl_methods["__call__"], args, kwargs)


def wrap_callback(f):
    return lambda *args, **kwargs: f(*args, **kwargs)
