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
        import numpy
        if self == Base.Bool:
            return numpy.dtype(numpy.bool_)
        if self == Base.Int8:
            return numpy.dtype(numpy.int8)
        if self == Base.Int16:
            return numpy.dtype(numpy.int16)
        if self == Base.Int32:
            return numpy.dtype(numpy.int32)
        if self == Base.Int64:
            return numpy.dtype(numpy.int64)
        if self == Base.Int:
            return numpy.dtype(numpy.int_)
        if self == Base.UInt8:
            return numpy.dtype(numpy.uint8)
        if self == Base.UInt16:
            return numpy.dtype(numpy.uint16)
        if self == Base.UInt32:
            return numpy.dtype(numpy.uint32)
        if self == Base.UInt64:
            return numpy.dtype(numpy.uint64)
        if self == Base.UInt:
            return numpy.dtype(numpy.uintp)
        if self == Base.Float16:
            return numpy.dtype(numpy.float16)
        if self == Base.Float32:
            return numpy.dtype(numpy.float32)
        if self == Base.Float64:
            return numpy.dtype(numpy.float64)
        if self == Base.ComplexF32:
            return numpy.dtype(numpy.complex64)
        if self == Base.ComplexF64:
            return numpy.dtype(numpy.complex128)
        if self == Base.Ptr[Base.Cvoid]:
            return numpy.dtype("P")
        raise AttributeError("__numpy_dtype__")
""",
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
    pycopy!(pyjltypetype, jl.TypeValue)
end

pyjltype(::Type) = pyjltypetype
