const pyjlnumbertype = pynew()
const pyjlcomplextype = pynew()
const pyjlrealtype = pynew()
const pyjlrationaltype = pynew()
const pyjlintegertype = pynew()

struct pyjlnumber_op{OP}
    op::OP
end
(op::pyjlnumber_op)(self) = Py(op.op(self))
function (op::pyjlnumber_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = @pyconvert(Number, other_, return pybuiltins.NotImplemented)
    end
    Py(op.op(self, other))
end
function (op::pyjlnumber_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = @pyconvert(Number, other_, return pybuiltins.NotImplemented)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        pydel!(other2_)
    else
        other2 = @pyconvert(Number, other2_, return pybuiltins.NotImplemented)
    end
    Py(op.op(self, other, other2))
end
pyjl_handle_error_type(op::pyjlnumber_op, self, exc) =
    exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

struct pyjlnumber_rev_op{OP}
    op::OP
end
function (op::pyjlnumber_rev_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = @pyconvert(Number, other_, return pybuiltins.NotImplemented)
    end
    Py(op.op(other, self))
end
function (op::pyjlnumber_rev_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = @pyconvert(Number, other_, return pybuiltins.NotImplemented)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        pydel!(other2_)
    else
        other2 = @pyconvert(Number, other2_, return pybuiltins.NotImplemented)
    end
    Py(op.op(other, self, other2))
end
pyjl_handle_error_type(op::pyjlnumber_rev_op, self, exc) =
    exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

pyjlreal_trunc(self::Real) = Py(trunc(Integer, self))
pyjl_handle_error_type(::typeof(pyjlreal_trunc), self, exc::MethodError) =
    exc.f === trunc ? pybuiltins.TypeError : PyNULL

pyjlreal_floor(self::Real) = Py(floor(Integer, self))
pyjl_handle_error_type(::typeof(pyjlreal_floor), self, exc::MethodError) =
    exc.f === floor ? pybuiltins.TypeError : PyNULL

pyjlreal_ceil(self::Real) = Py(ceil(Integer, self))
pyjl_handle_error_type(::typeof(pyjlreal_ceil), self, exc::MethodError) =
    exc.f === ceil ? pybuiltins.TypeError : PyNULL

function pyjlreal_round(self::Real, ndigits_::Py)
    ndigits = pyconvertarg(Union{Int,Nothing}, ndigits_, "ndigits")
    pydel!(ndigits_)
    if ndigits === nothing
        Py(round(Integer, self))
    else
        Py(round(self; digits = ndigits))
    end
end
pyjl_handle_error_type(::typeof(pyjlreal_round), self, exc::MethodError) =
    exc.f === round ? pybuiltins.TypeError : PyNULL

function init_number()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.number")
    methods = m._jl_methods
    methods["__bool__"] = pyjl_methodnum(pyjlnumber_op(iszero))
    methods["__add__"] = pyjl_methodnum(pyjlnumber_op(+))
    methods["__sub__"] = pyjl_methodnum(pyjlnumber_op(-))
    methods["__mul__"] = pyjl_methodnum(pyjlnumber_op(*))
    methods["__truediv__"] = pyjl_methodnum(pyjlnumber_op(/))
    methods["__floordiv__"] = pyjl_methodnum(pyjlnumber_op(÷))
    methods["__mod__"] = pyjl_methodnum(pyjlnumber_op(%))
    methods["__pow__"] = pyjl_methodnum(pyjlnumber_op(^))
    methods["__pow_modulo__"] = pyjl_methodnum(pyjlnumber_op(powermod))
    methods["__lshift__"] = pyjl_methodnum(pyjlnumber_op(<<))
    methods["__rshift__"] = pyjl_methodnum(pyjlnumber_op(>>))
    methods["__and__"] = pyjl_methodnum(pyjlnumber_op(&))
    methods["__xor__"] = pyjl_methodnum(pyjlnumber_op(⊻))
    methods["__or__"] = pyjl_methodnum(pyjlnumber_op(|))
    methods["__radd__"] = pyjl_methodnum(pyjlnumber_rev_op(+))
    methods["__rsub__"] = pyjl_methodnum(pyjlnumber_rev_op(-))
    methods["__rmul__"] = pyjl_methodnum(pyjlnumber_rev_op(*))
    methods["__rtruediv__"] = pyjl_methodnum(pyjlnumber_rev_op(/))
    methods["__rfloordiv__"] = pyjl_methodnum(pyjlnumber_rev_op(÷))
    methods["__rmod__"] = pyjl_methodnum(pyjlnumber_rev_op(%))
    methods["__rpow__"] = pyjl_methodnum(pyjlnumber_rev_op(^))
    methods["__rpow_modulo__"] = pyjl_methodnum(pyjlnumber_rev_op(powermod))
    methods["__rlshift__"] = pyjl_methodnum(pyjlnumber_rev_op(<<))
    methods["__rrshift__"] = pyjl_methodnum(pyjlnumber_rev_op(>>))
    methods["__rand__"] = pyjl_methodnum(pyjlnumber_rev_op(&))
    methods["__rxor__"] = pyjl_methodnum(pyjlnumber_rev_op(⊻))
    methods["__ror__"] = pyjl_methodnum(pyjlnumber_rev_op(|))
    methods["__eq__"] = pyjl_methodnum(pyjlnumber_op(==))
    methods["__ne__"] = pyjl_methodnum(pyjlnumber_op(!=))
    methods["__le__"] = pyjl_methodnum(pyjlnumber_op(≤))
    methods["__lt__"] = pyjl_methodnum(pyjlnumber_op(<))
    methods["__ge__"] = pyjl_methodnum(pyjlnumber_op(≥))
    methods["__gt__"] = pyjl_methodnum(pyjlnumber_op(>))
    methods["__complex__"] = pyjl_methodnum(pycomplex)
    methods["real"] = pyjl_methodnum(pyjlnumber_op(real))
    methods["imag"] = pyjl_methodnum(pyjlnumber_op(imag))
    methods["conjugate"] = pyjl_methodnum(pyjlnumber_op(conj))
    methods["__float__"] = pyjl_methodnum(pyfloat)
    methods["__trunc__"] = pyjl_methodnum(pyjlreal_trunc)
    methods["__floor__"] = pyjl_methodnum(pyjlreal_floor)
    methods["__ceil__"] = pyjl_methodnum(pyjlreal_ceil)
    methods["__round__"] = pyjl_methodnum(pyjlreal_round)
    methods["numerator"] = pyjl_methodnum(pyjlnumber_op(numerator))
    methods["denominator"] = pyjl_methodnum(pyjlnumber_op(denominator))
    methods["__int__"] = pyjl_methodnum(pyint)
    pycopy!(pyjlnumbertype, jl.NumberValue)
    pycopy!(pyjlcomplextype, jl.ComplexValue)
    pycopy!(pyjlrealtype, jl.RealValue)
    pycopy!(pyjlrationaltype, jl.RationalValue)
    pycopy!(pyjlintegertype, jl.IntegerValue)
end

pyjltype(::Number) = pyjlnumbertype
pyjltype(::Complex) = pyjlcomplextype
pyjltype(::Real) = pyjlrealtype
pyjltype(::Rational) = pyjlrationaltype
pyjltype(::Integer) = pyjlintegertype
