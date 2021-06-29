const pyjlarraytype = pynew()

function init_jlwrap_array()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class ArrayValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlarraytype, jl.ArrayValue)
end

pyjl(v::AbstractArray) = pyjl(pyjlarraytype, v)
