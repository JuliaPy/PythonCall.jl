"""Julia value wrappers from src/JlWrap/number.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class NumberValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __bool__(self):
        return not self._jl_callmethod(_jl_methods["__bool__"])
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
class ComplexValue(NumberValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __complex__(self):
        return self._jl_callmethod(_jl_methods["__complex__"])
    @property
    def real(self):
        return self._jl_callmethod(_jl_methods["real"])
    @property
    def imag(self):
        return self._jl_callmethod(_jl_methods["imag"])
    def conjugate(self):
        return self._jl_callmethod(_jl_methods["conjugate"])
class RealValue(ComplexValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __float__(self):
        return self._jl_callmethod(_jl_methods["__float__"])
    @property
    def real(self):
        return self
    @property
    def imag(self):
        return 0
    def conjugate(self):
        return self
    def __complex__(self):
        return complex(float(self))
    def __trunc__(self):
        return self._jl_callmethod(_jl_methods["__trunc__"])
    def __floor__(self):
        return self._jl_callmethod(_jl_methods["__floor__"])
    def __ceil__(self):
        return self._jl_callmethod(_jl_methods["__ceil__"])
    def __round__(self, ndigits=None):
        return self._jl_callmethod(_jl_methods["__round__"], ndigits)
class RationalValue(RealValue):
    __module__ = "juliacall"
    __slots__ = ()
    @property
    def numerator(self):
        return self._jl_callmethod(_jl_methods["numerator"])
    @property
    def denominator(self):
        return self._jl_callmethod(_jl_methods["denominator"])
class IntegerValue(RationalValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __int__(self):
        return self._jl_callmethod(_jl_methods["__int__"])
    def __index__(self):
        return self.__int__()
    @property
    def numerator(self):
        return self
    @property
    def denominator(self):
        return 1
import numbers
numbers.Number.register(NumberValue)
numbers.Complex.register(ComplexValue)
numbers.Real.register(RealValue)
numbers.Rational.register(RationalValue)
numbers.Integral.register(IntegerValue)
del numbers
