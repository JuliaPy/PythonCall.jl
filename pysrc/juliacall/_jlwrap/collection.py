"""Julia wrappers initialized by PythonCall."""

from juliacall import JlBase
from .any import JlBase2

# Populated by Julia during initialization.
_jl_methods = {}


class JlCollection(JlBase2):
    __module__ = "juliacall"
    __slots__ = ()
    def __len__(self):
        return self._jl_callmethod(_jl_methods["__len__"])
    def __bool__(self):
        return self._jl_callmethod(_jl_methods["__bool__"])
    def __iter__(self):
        return self._jl_callmethod(_jl_methods["__iter__"])
    def __hash__(self):
        return self._jl_callmethod(_jl_methods["__hash__"])
    def __eq__(self, other):
        if isinstance(self, type(other)) or isinstance(other, type(self)):
            return self._jl_callmethod(_jl_methods["__eq__"], other)
        else:
            return NotImplemented
    def __contains__(self, v):
        return self._jl_callmethod(_jl_methods["__contains__"], v)
    def copy(self):
        return self._jl_callmethod(_jl_methods["copy"])
    def clear(self):
        return self._jl_callmethod(_jl_methods["clear"])
import collections.abc
collections.abc.Collection.register(JlCollection)
del collections
