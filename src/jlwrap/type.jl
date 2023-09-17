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

function init_type()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class TypeValue(AnyValue):
        __slots__ = ()
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjltype_getitem)), k)
        def __setitem__(self, k, v):
            raise TypeError("not supported")
        def __delitem__(self, k):
            raise TypeError("not supported")
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjltype(::Type) = pyjltypetype
