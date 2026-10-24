"""Julia value wrappers from src/JlWrap/type.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class TypeValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __getitem__(self, k):
        return self._jl_callmethod(_jl_methods["__getitem__"], k)
    def __setitem__(self, k, v):
        raise TypeError("not supported")
    def __delitem__(self, k):
        raise TypeError("not supported")
    @property
    def __numpy_dtype__(self):
        return self._jl_callmethod(_jl_methods["__numpy_dtype__"])
