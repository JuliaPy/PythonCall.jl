pybytes_fromdata(x::Ptr, n::Integer) = setptr!(pynew(), errcheck(C.PyBytes_FromStringAndSize(x, n)))
pybytes_fromdata(x) = pybytes_fromdata(pointer(x), sizeof(x))

pybytes(x) = setptr!(pynew(), errcheck(@autopy x C.PyObject_Bytes(getptr(x_))))
pybytes(x::Vector{UInt8}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8, String}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8, SubString{String}}) = pybytes_fromdata(x)
pybytes(::Type{T}, x) where {Vector{UInt8} <: T <: Vector} = (b=pybytes(x); ans=pybytes_asvector(b); pydone!(b); ans)
pybytes(::Type{T}, x) where {Base.CodeUnits{UInt8,String} <: T <: Base.CodeUnits} = (b=pybytes(x); ans=Base.CodeUnits(pybytes_asUTF8string(b)); pydone!(b); ans)
export pybytes

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
