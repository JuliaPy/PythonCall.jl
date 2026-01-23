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
    np = pyimport("numpy")
    if self === Bool
        return np.dtype(np.bool_)
    elseif self === Int8
        return np.dtype(np.int8)
    elseif self === Int16
        return np.dtype(np.int16)
    elseif self === Int32
        return np.dtype(np.int32)
    elseif self === Int64
        return np.dtype(np.int64)
    elseif self === UInt8
        return np.dtype(np.uint8)
    elseif self === UInt16
        return np.dtype(np.uint16)
    elseif self === UInt32
        return np.dtype(np.uint32)
    elseif self === UInt64
        return np.dtype(np.uint64)
    elseif self === Float16
        return np.dtype(np.float16)
    elseif self === Float32
        return np.dtype(np.float32)
    elseif self === Float64
        return np.dtype(np.float64)
    elseif self === ComplexF32
        return np.dtype(np.complex64)
    elseif self === ComplexF64
        return np.dtype(np.complex128)
    elseif self === Ptr{Cvoid}
        return np.dtype("P")
    end
    errset(pybuiltins.AttributeError, "__numpy_dtype__")
    return PyNULL
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
