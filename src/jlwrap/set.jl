const pyjlsettype = pynew()

function init_jlwrap_set()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class SetValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlsettype, jl.SetValue)
end

pyjl(v::AbstractSet) = pyjl(pyjlsettype, v)
