const pyjltypetype = pynew()

function pyjltype_getitem(self::Type, k_)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        Py(self{k...})
    else
        k = pyconvert(Any, k_)
        Py(self{k})
    end
end

function init_jlwrap_type()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class TypeValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjltype_getitem)), k)
        def __setitem__(self, k, v):
            raise TypeError("not supported")
        def __delitem__(self, k):
            raise TypeError("not supported")
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjl(v::Type) = pyjl(pyjltypetype, v)
