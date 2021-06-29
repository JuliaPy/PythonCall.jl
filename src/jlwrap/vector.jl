const pyjlvectortype = pynew()

function init_jlwrap_vector()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class VectorValue(ArrayValue):
        __slots__ = ()
        __module__ = "juliacall"
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlvectortype, jl.VectorValue)
end

pyjl(v::AbstractVector) = pyjl(pyjlvectortype, v)
