"""Julia value wrappers from src/JlWrap/set.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class SetValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __bool__(self):
        return bool(len(self))
    def add(self, value):
        return self._jl_callmethod(_jl_methods["add"], value)
    def discard(self, value):
        return self._jl_callmethod(_jl_methods["discard"], value)
    def clear(self):
        return self._jl_callmethod(_jl_methods["clear"])
    def copy(self):
        return self._jl_callmethod(_jl_methods["copy"])
    def pop(self):
        return self._jl_callmethod(_jl_methods["pop"])
    def remove(self, value):
        return self._jl_callmethod(_jl_methods["remove"], value)
    def difference(self, other):
        return set(self).difference(other)
    def intersection(self, other):
        return set(self).intersection(other)
    def symmetric_difference(self, other):
        return set(self).symmetric_difference(other)
    def union(self, other):
        return set(self).union(other)
    def isdisjoint(self, other):
        return set(self).isdisjoint(other)
    def issubset(self, other):
        return set(self).issubset(other)
    def issuperset(self, other):
        return set(self).issuperset(other)
    def difference_update(self, other):
        return self._jl_callmethod(_jl_methods["difference_update"], other)
    def intersection_update(self, other):
        return self._jl_callmethod(_jl_methods["intersection_update"], other)
    def symmetric_difference_update(self, other):
        return self._jl_callmethod(_jl_methods["symmetric_difference_update"], other)
    def update(self, other):
        return self._jl_callmethod(_jl_methods["update"], other)
import collections.abc
collections.abc.MutableSet.register(SetValue)
del collections
