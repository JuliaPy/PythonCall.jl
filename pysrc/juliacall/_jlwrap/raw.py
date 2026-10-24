"""Julia value wrappers from src/JlWrap/raw.jl."""

from juliacall import ValueBase

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class RawValue(ValueBase):
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
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        else:
            return self._jl_callmethod(_jl_methods["__getattr__"], k)
    def __setattr__(self, k, v):
        try:
            ValueBase.__setattr__(self, k, v)
        except AttributeError:
            if k.startswith("__") and k.endswith("__"):
                raise
        else:
            return
        self._jl_callmethod(_jl_methods["__setattr__"], k, v)
    def __dir__(self):
        return ValueBase.__dir__(self) + self._jl_callmethod(_jl_methods["__dir__"])
    def __call__(self, *args, **kwargs):
        return self._jl_callmethod(_jl_methods["__call__"], args, kwargs)
    def __len__(self):
        return self._jl_callmethod(_jl_methods["__len__"])
    def __getitem__(self, k):
        return self._jl_callmethod(_jl_methods["__getitem__"], k)
    def __setitem__(self, k, v):
        self._jl_callmethod(_jl_methods["__setitem__"], k, v)
    def __delitem__(self, k):
        self._jl_callmethod(_jl_methods["__delitem__"], k)
    def __bool__(self):
        return self._jl_callmethod(_jl_methods["__bool__"])
    def _jl_any(self):
        '''Convert this to a juliacall.AnyValue.'''
        return self._jl_callmethod(_jl_methods["_jl_any"])
    def _jl_call_nogil(self, *args, **kwargs):
        '''Call this with the given arguments but with the GIL disabled.

        WARNING: This function must not interact with Python at all without re-acquiring
        the GIL.
        '''
        return self._jl_callmethod(_jl_methods["_jl_call_nogil"], args, kwargs)
