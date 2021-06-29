const pyjldicttype = pynew()

function init_jlwrap_dict()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class DictValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjldicttype, jl.DictValue)
end

pyjl(v::AbstractDict) = pyjl(pyjldicttype, v)
