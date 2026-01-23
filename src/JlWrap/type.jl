const pyjltypetype = pynew()

function pyjltype_getitem(self::Type, k_)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        Py(self{k...})
    else
        k = pyconvert(Any, k_)
        Py(self{k})
    end
end

function pyjltype_numpy_dtype(self::Type)
    typestr, descr = pytypestrdescr(self)
    if isempty(typestr)
        errset(pybuiltins.AttributeError, "__numpy_dtype__")
        return PyNULL
    end
    np = pyimport("numpy")
    if pyisnull(descr)
        return np.dtype(typestr)
    else
        return np.dtype(descr)
    end
end

pyjl_handle_error_type(::typeof(pyjltype_numpy_dtype), x, exc) = pybuiltins.AttributeError

function init_type()
    jl = pyjuliacallmodule
    pybuiltins.exec(
        pybuiltins.compile(
            """
$("\n"^(@__LINE__()-1))
class TypeValue(AnyValue):
    __slots__ = ()
    def __getitem__(self, k):
        return self._jl_callmethod($(pyjl_methodnum(pyjltype_getitem)), k)
    def __setitem__(self, k, v):
        raise TypeError("not supported")
    def __delitem__(self, k):
        raise TypeError("not supported")
    @property
    def __numpy_dtype__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjltype_numpy_dtype)))
""",
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjltype(::Type) = pyjltypetype
