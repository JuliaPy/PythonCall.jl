pybytes_fromdata(x::Ptr, n::Integer) = pynew(errcheck(C.PyBytes_FromStringAndSize(x, n)))
pybytes_fromdata(x) = pybytes_fromdata(pointer(x), sizeof(x))

pybytes(x) = pynew(errcheck(@autopy x C.PyObject_Bytes(getptr(x_))))
pybytes(x::Vector{UInt8}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8, String}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8, SubString{String}}) = pybytes_fromdata(x)
pybytes(::Type{T}, x) where {Vector{UInt8} <: T <: Vector} = (b=pybytes(x); ans=pybytes_asvector(b); pydel!(b); ans)
pybytes(::Type{T}, x) where {Base.CodeUnits{UInt8,String} <: T <: Base.CodeUnits} = (b=pybytes(x); ans=Base.CodeUnits(pybytes_asUTF8string(b)); pydel!(b); ans)
export pybytes

pyisbytes(x) = pytypecheckfast(x, C.Py_TPFLAGS_BYTES_SUBCLASS)

function pybytes_asdata(x::Py)
    ptr = Ref(Ptr{Cchar}(0))
    len = Ref(C.Py_ssize_t(0))
    errcheck(C.PyBytes_AsStringAndSize(getptr(x), ptr, len))
    ptr[], len[]
end

function pybytes_asvector(x::Py)
    ptr, len = pybytes_asdata(x)
    unsafe_wrap(Array, Ptr{UInt8}(ptr), len)
end

function pybytes_asUTF8string(x::Py)
    ptr, len = pybytes_asdata(x)
    unsafe_string(Ptr{UInt8}(ptr), len)
end

pyconvert_rule_bytes(::Type{Vector{UInt8}}, x::Py) = pyconvert_return(copy(pybytes_asvector(x)))
pyconvert_rule_bytes(::Type{Base.CodeUnits{UInt8,String}}, x::Py) = pyconvert_return(codeunits(pybytes_asUTF8string(x)))

pyconvert_rule_fast(::Type{Vector{UInt8}}, x::Py) = pyisbytes(x) ? pyconvert_rule_bytes(Vector{UInt8}, x) : pyconvert_unconverted()
pyconvert_rule_fast(::Type{Base.CodeUnits{UInt8,String}}, x::Py) = pyisbytes(x) ? pyconvert_rule_bytyes(Base.CodeUnits{UInt8,String}, x) : pyconvert_unconverted()
