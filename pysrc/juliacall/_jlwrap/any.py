"""Julia wrappers initialized by PythonCall."""

from juliacall import JlBase

# Populated by Julia during initialization.
_jl_methods = {}


class JlBase2(JlBase):
    __module__ = "juliacall"
    __slots__ = ()
    def __repr__(self):
        t = type(self)
        if t is Jl:
            name = "Julia"
        else:
            name = t.__name__
        return name + ":" + self._jl_callmethod(_jl_methods["__repr__"])
class JlIter(JlBase2):
    __module__ = "juliacall"
    __slots__ = ()
    def __iter__(self):
        return self
    def __hash__(self):
        return self._jl_callmethod(_jl_methods["__hash__"])
    def __next__(self):
        return self._jl_callmethod(_jl_methods["iterator_next"])
class Jl(JlBase2):
    __module__ = "juliacall"
    __slots__ = ()
    def __str__(self):
        return self._jl_callmethod(_jl_methods["__str__"])
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        else:
            return self._jl_callmethod(_jl_methods["__getattr__"], k)
    def __setattr__(self, k, v):
        try:
            JlBase.__setattr__(self, k, v)
        except AttributeError:
            if k.startswith("__") and k.endswith("__"):
                raise
        else:
            return
        self._jl_callmethod(_jl_methods["__setattr__"], k, v)
    def __dir__(self):
        return JlBase.__dir__(self) + self._jl_callmethod(_jl_methods["__dir__"])
    def __call__(self, *args, **kwargs):
        return self._jl_callmethod(_jl_methods["__call__"], args, kwargs)
    def __bool__(self):
        return self._jl_callmethod(_jl_methods["__bool__"])
    def __len__(self):
        return self._jl_callmethod(_jl_methods["__len__"])
    def __getitem__(self, k):
        return self._jl_callmethod(_jl_methods["__getitem__"], k)
    def __setitem__(self, k, v):
        self._jl_callmethod(_jl_methods["__setitem__"], k, v)
    def __delitem__(self, k):
        self._jl_callmethod(_jl_methods["__delitem__"], k)
    def __iter__(self):
        return self._jl_callmethod(_jl_methods["__iter__"])
    def __next__(self):
        return self._jl_callmethod(_jl_methods["__next__"])
    def __reversed__(self):
        return self._jl_callmethod(_jl_methods["__reversed__"])
    def __contains__(self, v):
        return self._jl_callmethod(_jl_methods["__contains__"], v)
    def __pos__(self):
        return self._jl_callmethod(_jl_methods["__pos__"])
    def __neg__(self):
        return self._jl_callmethod(_jl_methods["__neg__"])
    def __abs__(self):
        return self._jl_callmethod(_jl_methods["__abs__"])
    def abs(self):
        return self._jl_callmethod(_jl_methods["__abs__"])
    def __invert__(self):
        return self._jl_callmethod(_jl_methods["__invert__"])
    def __add__(self, other):
        return self._jl_callmethod(_jl_methods["__add__"], other)
    def __sub__(self, other):
        return self._jl_callmethod(_jl_methods["__sub__"], other)
    def __mul__(self, other):
        return self._jl_callmethod(_jl_methods["__mul__"], other)
    def __truediv__(self, other):
        return self._jl_callmethod(_jl_methods["__truediv__"], other)
    def __floordiv__(self, other):
        return self._jl_callmethod(_jl_methods["__floordiv__"], other)
    def __mod__(self, other):
        return self._jl_callmethod(_jl_methods["__mod__"], other)
    def __pow__(self, other, modulo=None):
        if modulo is None:
            return self._jl_callmethod(_jl_methods["__pow__"], other)
        else:
            return self._jl_callmethod(_jl_methods["__pow_modulo__"], other, modulo)
    def __lshift__(self, other):
        return self._jl_callmethod(_jl_methods["__lshift__"], other)
    def __rshift__(self, other):
        return self._jl_callmethod(_jl_methods["__rshift__"], other)
    def __and__(self, other):
        return self._jl_callmethod(_jl_methods["__and__"], other)
    def __xor__(self, other):
        return self._jl_callmethod(_jl_methods["__xor__"], other)
    def __or__(self, other):
        return self._jl_callmethod(_jl_methods["__or__"], other)
    def __radd__(self, other):
        return self._jl_callmethod(_jl_methods["__radd__"], other)
    def __rsub__(self, other):
        return self._jl_callmethod(_jl_methods["__rsub__"], other)
    def __rmul__(self, other):
        return self._jl_callmethod(_jl_methods["__rmul__"], other)
    def __rtruediv__(self, other):
        return self._jl_callmethod(_jl_methods["__rtruediv__"], other)
    def __rfloordiv__(self, other):
        return self._jl_callmethod(_jl_methods["__rfloordiv__"], other)
    def __rmod__(self, other):
        return self._jl_callmethod(_jl_methods["__rmod__"], other)
    def __rpow__(self, other, modulo=None):
        if modulo is None:
            return self._jl_callmethod(_jl_methods["__rpow__"], other)
        else:
            return self._jl_callmethod(_jl_methods["__rpow_modulo__"], other, modulo)
    def __rlshift__(self, other):
        return self._jl_callmethod(_jl_methods["__rlshift__"], other)
    def __rrshift__(self, other):
        return self._jl_callmethod(_jl_methods["__rrshift__"], other)
    def __rand__(self, other):
        return self._jl_callmethod(_jl_methods["__rand__"], other)
    def __rxor__(self, other):
        return self._jl_callmethod(_jl_methods["__rxor__"], other)
    def __ror__(self, other):
        return self._jl_callmethod(_jl_methods["__ror__"], other)
    def __hash__(self):
        return self._jl_callmethod(_jl_methods["__hash__"])
    def __eq__(self, other):
        return self._jl_callmethod(_jl_methods["__eq__"], other)
    def __ne__(self, other):
        return self._jl_callmethod(_jl_methods["__ne__"], other)
    def __le__(self, other):
        return self._jl_callmethod(_jl_methods["__le__"], other)
    def __lt__(self, other):
        return self._jl_callmethod(_jl_methods["__lt__"], other)
    def __ge__(self, other):
        return self._jl_callmethod(_jl_methods["__ge__"], other)
    def __gt__(self, other):
        return self._jl_callmethod(_jl_methods["__gt__"], other)
    def __int__(self):
        return self._jl_callmethod(_jl_methods["__int__"])
    def __float__(self):
        return self._jl_callmethod(_jl_methods["__float__"])
    def __complex__(self):
        return self._jl_callmethod(_jl_methods["__complex__"])
    def __index__(self):
        return self._jl_callmethod(_jl_methods["__index__"])
    def __trunc__(self):
        return self._jl_callmethod(_jl_methods["__trunc__"])
    def __floor__(self):
        return self._jl_callmethod(_jl_methods["__floor__"])
    def __ceil__(self):
        return self._jl_callmethod(_jl_methods["__ceil__"])
    def __round__(self, ndigits=None):
        if ndigits is None:
            return self._jl_callmethod(_jl_methods["__round__"])
        else:
            return self._jl_callmethod(_jl_methods["__round__"], ndigits)
    @property
    def __name__(self):
        return self._jl_callmethod(_jl_methods["__name__"])
    def jl_display(self, mime=None):
        '''Display this, optionally specifying the MIME type.'''
        return self._jl_callmethod(_jl_methods["jl_display"], mime)
    def jl_help(self, mime=None):
        '''Show help for this Julia object.'''
        return self._jl_callmethod(_jl_methods["jl_help"], mime)
    def jl_eval(self, expr):
        return self._jl_callmethod(_jl_methods["jl_eval"], expr)
    def jl_to_py(self):
        return self._jl_callmethod(_jl_methods["jl_to_py"])
    def jl_callback(self, *args, **kwargs):
        return self._jl_callmethod(_jl_methods["jl_callback"], args, kwargs)
    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._jl_callmethod(_jl_methods["_repr_mimebundle_"], include, exclude)
    @property
    def __numpy_dtype__(self):
        return self._jl_callmethod(_jl_methods["__numpy_dtype__"])
