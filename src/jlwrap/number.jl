const pyjlnumbertype = pynew()
const pyjlcomplextype = pynew()
const pyjlrealtype = pynew()
const pyjlrationaltype = pynew()
const pyjlintegertype = pynew()

function init_jlwrap_number()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class NumberValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
    class ComplexValue(NumberValue):
        __slots__ = ()
        __module__ = "juliacall"
    class RealValue(ComplexValue):
        __slots__ = ()
        __module__ = "juliacall"
    class RationalValue(RealValue):
        __slots__ = ()
        __module__ = "juliacall"
    class IntegerValue(RationalValue):
        __slots__ = ()
        __module__ = "juliacall"
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
