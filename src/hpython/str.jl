"""
    pystr_decodeutf8(ctx, x)

Convert `x` to a Python string, assuming that `pointer(x)` points to a UTF8 encoded string of length `sizeof(x)`.

This works at least for `String`, `SubString{String}` and `Vector{UInt8}`.
"""
pystr_decodeutf8(ctx::Context, x::Ptr, len::Integer) = ctx.newhdl(ctx._c.PyUnicode_DecodeUTF8(x, len, C_NULL))
pystr_decodeutf8(ctx::Context, x) = pystr_decodeutf8(ctx, pointer(x), sizeof(x))
(f::Builtin{:str_decodeutf8})(args...) = pystr_decodeutf8(f.ctx, args...)

"""
    pystr_encodeutf8(ctx, x)

Encode the Python string `x` as UTF8 `bytes`.
"""
function pystr_encodeutf8(ctx::Context, x::PyAnyHdl)
    ans = PyNULL
    @autohdl ctx x
    ans = ctx.newhdl(ctx._c.PyUnicode_AsUTF8String(ctx.cptr(x)))
    @autoclosehdl ctx x
    return ans
end
(f::Builtin{:str_encodeutf8})(args...) = pystr_encodeutf8(f.ctx, args...)

"""
    pystr([T], ctx, x)

Convert `x` to a Python string. Optionally convert that to a `T`.
"""
pystr(ctx::Context, x::Union{String,SubString{String}}) = pystr_decodeutf8(ctx, x)
pystr(ctx::Context, x::AbstractString) = pystr(convert(String, x))
function pystr(ctx::Context, x)
    ans = PyNULL
    @autohdl ctx x
    ans = ctx.newhdl(ctx._c.PyObject_Str(ctx.cptr(x)))
    @autoclosehdl ctx x
    return ans
end
pystr(::Type{T}, ctx::Context, x) where {T} = pystr_convert(T, ctx, pystr(ctx, x).auto)
(f::Builtin{:str})(x) = pystr(f.ctx, x)
(f::Builtin{:str})(::Type{T}, x) where {T} = pystr(T, f.ctx, x)

"""
    pystr_convert(T, ctx, x)

Convert Python string `x` to a `T`.

Allowed types are:
- `String`: On error, returns an empty string.
- `Vector{UInt8}` or `Vector`: This does a no-copy conversion, so is only valid while the bytes object is alive. On error, returns an empty array.
- `Symbol`: On error, returns `Symbol()`.
- `Char`: On error, returns `Char(0) = '\\0'`.
"""
function pystr_convert(::Type{T}, ctx::Context, x::PyAnyHdl) where {T}
    ctx.bytes_convert(T, ctx.str_encodeutf8(x).auto)
end
(f::Builtin{:str_convert})(::Type{T}, x) where {T} = pystr_convert(T, f.ctx, x)
