"""Julia wrappers initialized by PythonCall."""

import juliacall as jl
from juliacall import JlBase
from .collection import JlCollection

# Populated by Julia during initialization.
_jl_methods = {}


class JlDict(JlCollection):
    __module__ = "juliacall"
    __slots__ = ()
    _jl_undefined_ = object()
    def __init__(self, value=None):
        if value is None:
            value = jl.Base.Dict()
        JlBase.__init__(self, value, jl.Base.AbstractDict)
    def __iter__(self):
        return self._jl_callmethod(_jl_methods["__iter__"])
    def __contains__(self, key):
        return self._jl_callmethod(_jl_methods["__contains__"], key)
    def __getitem__(self, key):
        if key in self:
            return self._jl_callmethod(_jl_methods["__getitem__"], key)
        else:
            raise KeyError(key)
    def __setitem__(self, key, value):
        return self._jl_callmethod(_jl_methods["__setitem__"], key, value)
    def __delitem__(self, key):
        if key in self:
            return self._jl_callmethod(_jl_methods["__delitem__"], key)
        else:
            raise KeyError(key)
    def keys(self):
        return self._jl_callmethod(_jl_methods["keys"])
    def values(self):
        return self._jl_callmethod(_jl_methods["values"])
    def items(self):
        return self._jl_callmethod(_jl_methods["items"])
    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default
    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]
    def pop(self, key, default=_jl_undefined_):
        if key in self:
            ans = self[key]
            del self[key]
            return ans
        elif default is self._jl_undefined_:
            raise KeyError(key)
        else:
            return default
    def popitem(self):
        if len(self):
            return self._jl_callmethod(_jl_methods["popitem"])
        else:
            raise KeyError()
    def update(self, other=_jl_undefined_, **kwargs):
        if other is self._jl_undefined_:
            pass
        else:
            if hasattr(other, "keys"):
                items = ((k, other[k]) for k in other.keys())
            else:
                items = other
            self._jl_callmethod(_jl_methods["update"], items)
        if kwargs:
            self.update(kwargs)
import collections.abc
collections.abc.MutableMapping.register(JlDict)
del collections
