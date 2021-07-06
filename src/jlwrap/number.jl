const pyjlnumbertype = pynew()
const pyjlcomplextype = pynew()
const pyjlrealtype = pynew()
const pyjlrationaltype = pynew()
const pyjlintegertype = pynew()

struct pyjlnumber_op{OP}
    op :: OP
end
function (op::pyjlnumber_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
    else
        r = pytryconvert(Number, other_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other = pyconvert_result(Number, r)
    end
    Py(op.op(self, other))
end
function (op::pyjlnumber_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
    else
        r = pytryconvert(Number, other_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other = pyconvert_result(Number, r)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
    else
        r = pytryconvert(Number, other2_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other2 = pyconvert_result(Number, r)
    end
    Py(op.op(self, other, other2))
end
pyjl_handle_error_type(op::pyjlnumber_op, self, exc) = exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

struct pyjlnumber_rev_op{OP}
    op :: OP
end
function (op::pyjlnumber_rev_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
    else
        r = pytryconvert(Number, other_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other = pyconvert_result(Number, r)
    end
    Py(op.op(other, self))
end
function (op::pyjlnumber_rev_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
    else
        r = pytryconvert(Number, other_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other = pyconvert_result(Number, r)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
    else
        r = pytryconvert(Number, other2_)
        pyconvert_isunconverted(r) && return Py(pybuiltins.NotImplemented)
        other2 = pyconvert_result(Number, r)
    end
    Py(op.op(other, self, other2))
end
pyjl_handle_error_type(op::pyjlnumber_rev_op, self, exc) = exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

function init_jlwrap_number()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class NumberValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __add__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(+))), other)
        def __sub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(-))), other)
        def __mul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(*))), other)
        def __truediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(/))), other)
        def __floordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(÷))), other)
        def __mod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(%))), other)
        def __pow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(powermod))), other, modulo)
        def __lshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(<<))), other)
        def __rshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(>>))), other)
        def __and__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(&))), other)
        def __xor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(⊻))), other)
        def __or__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_op(|))), other)
        def __radd__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(+))), other)
        def __rsub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(-))), other)
        def __rmul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(*))), other)
        def __rtruediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(/))), other)
        def __rfloordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(÷))), other)
        def __rmod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(%))), other)
        def __rpow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(powermod))), other, modulo)
        def __rlshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(<<))), other)
        def __rrshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(>>))), other)
        def __rand__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(&))), other)
        def __rxor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(⊻))), other)
        def __ror__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlnumber_rev_op(|))), other)
    class ComplexValue(NumberValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __complex__(self):
            return self._jl_callmethod($(pyjl_methodnum(pycomplex)))
    class RealValue(ComplexValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __float__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyfloat)))
    class RationalValue(RealValue):
        __slots__ = ()
        __module__ = "juliacall"
    class IntegerValue(RationalValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __int__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyint)))
        def __index__(self):
            return self.__int__()
    import numbers
    numbers.Number.register(NumberValue)
    numbers.Complex.register(ComplexValue)
    numbers.Real.register(RealValue)
    numbers.Rational.register(RationalValue)
    numbers.Integral.register(IntegerValue)
    del numbers
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlnumbertype, jl.NumberValue)
    pycopy!(pyjlcomplextype, jl.ComplexValue)
    pycopy!(pyjlrealtype, jl.RealValue)
    pycopy!(pyjlrationaltype, jl.RationalValue)
    pycopy!(pyjlintegertype, jl.IntegerValue)
end

pyjl(v::Number) = pyjl(pyjlnumbertype, v)
pyjl(v::Complex) = pyjl(pyjlcomplextype, v)
pyjl(v::Real) = pyjl(pyjlrealtype, v)
pyjl(v::Rational) = pyjl(pyjlrationaltype, v)
pyjl(v::Integer) = pyjl(pyjlintegertype, v)
