"""
    pybytes_fromdata(ctx, x)

Construct a Python `bytes` from `sizeof(x)` bytes at `pointer(x)`.
"""
pybytes_fromdata(ctx::Context, ptr::Ptr, len::Integer) = ctx.newhdl(ctx._c.PyBytes_FromStringAndSize(ptr, len))
pybytes_fromdata(ctx::Context, x) = pybytes_fromdata(ctx, pointer(x), sizeof(x))
(f::Builtin{:bytes_fromdata})(args...) = pybytes_fromdata(f.ctx, args...)

function pybytes(ctx::Context, x::Union{String,SubString{String},Vector{UInt8}})
    pybytes_fromdata(ctx, x)
end
function pybytes(ctx::Context, x)
    ans = PyNULL
    @autohdl ctx x
    ans = ctx.newhdl(ctx._c.PyObject_Bytes(ctx.cptr(x)))
    @autoclosehdl ctx x
    ans
end
pybytes(::Type{T}, ctx::Context, x) where {T} = pybytes_convert(T, ctx, pybytes(ctx, x).auto)
(f::Builtin{:bytes})(x) = pybytes(f.ctx, x)
(f::Builtin{:bytes})(::Type{T}, x) where {T} = pybytes(T, f.ctx, x)

"""
    pybytes_aspointer(ctx, x)

Given a Python `bytes` object `x`, returns a pointer to its data and its length.

On error, the pointer is NULL and the length is 0.
"""
function pybytes_aspointer(ctx::Context, x::PyAnyHdl)
    err = Cint(-1)
    @autohdl ctx x
    ptr = Ref(C_NULL)
    len = Ref(C.Py_ssize_t(0))
    err = ctx._c.PyBytes_AsStringAndSize(ctx.cptr(x), ptr, len)
    @autoclosehdl ctx x
    err == -1 ? (C_NULL, 0) : (ptr[], len[])
end
(f::Builtin{:bytes_aspointer})(args...) = pybytes_aspointer(f.ctx, args...)

"""
    pybytes_convert(T, ctx, x)

Convert the Python bytes object to a `T`.

Allowed types are:
- `String`: On error, returns an empty string.
- `Vector{UInt8}` or `Vector`: This does a no-copy conversion, so is only valid while the bytes object is alive. On error, returns an empty array.
- `Symbol`: On error, returns `Symbol()`.
- `Char`: On error, returns `Char(0) = '\\0'`.
"""
function pybytes_convert(::Type{String}, ctx::Context, x::PyAnyHdl)
    ptr, len = pybytes_aspointer(ctx, x)
    ptr == C_NULL ? "" : Base.unsafe_string(Ptr{UInt8}(ptr), len)
end
function pybytes_convert(::Union{Type{Vector{UInt8}},Type{Vector}}, ctx::Context, x::PyAnyHdl)
    ptr, len = pybytes_aspointer(ctx, x)
    ptr == C_NULL ? UInt8[] : Base.unsafe_wrap(Array, Ptr{UInt8}(ptr), len)
end
function pybytes_convert(::Type{Symbol}, ctx::Context, x::PyAnyHdl)
    Symbol(pybytes_convert(String, ctx, x))
end
function pybytes_convert(::Type{Char}, ctx::Context, x::PyAnyHdl)
    str = pybytes_convert(String, ctx, x)
    if length(str) == 1
        first(str)
    elseif isempty(str) && iserr(ctx)
        Char(0)
    else
        ctx.errset(ctx.ValueError, "expecting a single character")
        Char(0)
    end
end
(f::Builtin{:bytes_convert})(::Type{T}, x) where {T} = pybytes_convert(T, f.ctx, x)
