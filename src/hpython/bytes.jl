"""
    pybytes_fromdata(py, x)

Construct a Python `bytes` from `sizeof(x)` bytes at `pointer(x)`.
"""
pybytes_fromdata(py::Context, ptr::Ptr, len::Integer) = py.newhdl(py._c.PyBytes_FromStringAndSize(ptr, len))
pybytes_fromdata(py::Context, x) = pybytes_fromdata(py, pointer(x), sizeof(x))
(f::Builtin{:bytes_fromdata})(args...) = pybytes_fromdata(f.ctx, args...)

function pybytes(py::Context, x::Union{String,SubString{String},Vector{UInt8}})
    pybytes_fromdata(py, x)
end
function pybytes(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyObject_Bytes(py.cptr(x)))
    @autoclosehdl py x
    ans
end
pybytes(::Type{T}, py::Context, x) where {T} = pybytes_convert(T, py, pybytes(py, x).auto)
(f::Builtin{:bytes})(x) = pybytes(f.ctx, x)
(f::Builtin{:bytes})(::Type{T}, x) where {T} = pybytes(T, f.ctx, x)

"""
    pybytes_aspointer(py, x)

Given a Python `bytes` object `x`, returns a pointer to its data and its length.

On error, the pointer is NULL and the length is 0.
"""
function pybytes_aspointer(py::Context, x::PyAnyHdl)
    err = Cint(-1)
    @autohdl py x
    ptr = Ref(C_NULL)
    len = Ref(C.Py_ssize_t(0))
    err = py._c.PyBytes_AsStringAndSize(py.cptr(x), ptr, len)
    @autoclosehdl py x
    err == -1 ? (C_NULL, 0) : (ptr[], len[])
end
(f::Builtin{:bytes_aspointer})(args...) = pybytes_aspointer(f.ctx, args...)

"""
    pybytes_convert(T, py, x)

Convert the Python bytes object to a `T`.

Allowed types are:
- `String`: On error, returns an empty string.
- `Vector{UInt8}` or `Vector`: This does a no-copy conversion, so is only valid while the bytes object is alive. On error, returns an empty array.
- `Symbol`: On error, returns `Symbol()`.
- `Char`: On error, returns `Char(0) = '\\0'`.
"""
function pybytes_convert(::Type{String}, py::Context, x::PyAnyHdl)
    ptr, len = pybytes_aspointer(py, x)
    ptr == C_NULL ? "" : Base.unsafe_string(Ptr{UInt8}(ptr), len)
end
function pybytes_convert(::Union{Type{Vector{UInt8}},Type{Vector}}, py::Context, x::PyAnyHdl)
    ptr, len = pybytes_aspointer(py, x)
    ptr == C_NULL ? UInt8[] : Base.unsafe_wrap(Array, Ptr{UInt8}(ptr), len)
end
function pybytes_convert(::Type{Symbol}, py::Context, x::PyAnyHdl)
    Symbol(pybytes_convert(String, py, x))
end
function pybytes_convert(::Type{Char}, py::Context, x::PyAnyHdl)
    str = pybytes_convert(String, py, x)
    if length(str) == 1
        first(str)
    elseif isempty(str) && iserr(py)
        Char(0)
    else
        py.errset(py.ValueError, "expecting a single character")
        Char(0)
    end
end
(f::Builtin{:bytes_convert})(::Type{T}, x) where {T} = pybytes_convert(T, f.ctx, x)
