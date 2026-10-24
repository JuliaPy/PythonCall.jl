"""Julia value wrappers from src/JlWrap/vector.jl."""

from .array import ArrayValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class VectorValue(ArrayValue):
    __module__ = "juliacall"
    __slots__ = ()
    def resize(self, size):
        return self._jl_callmethod(_jl_methods["resize"], size)
    def sort(self, reverse=False, key=None):
        return self._jl_callmethod(_jl_methods["sort"], reverse, key)
    def reverse(self):
        return self._jl_callmethod(_jl_methods["reverse"])
    def clear(self):
        return self._jl_callmethod(_jl_methods["clear"])
    def __reversed__(self):
        return self._jl_callmethod(_jl_methods["__reversed__"])
    def insert(self, index, value):
        return self._jl_callmethod(_jl_methods["insert"], index, value)
    def append(self, value):
        return self._jl_callmethod(_jl_methods["append"], value)
    def extend(self, values):
        return self._jl_callmethod(_jl_methods["extend"], values)
    def pop(self, index=-1):
        return self._jl_callmethod(_jl_methods["pop"], index)
    def remove(self, value):
        return self._jl_callmethod(_jl_methods["remove"], value)
    def index(self, value):
        return self._jl_callmethod(_jl_methods["index"], value)
    def count(self, value):
        return self._jl_callmethod(_jl_methods["count"], value)
import collections.abc
collections.abc.MutableSequence.register(VectorValue)
del collections
