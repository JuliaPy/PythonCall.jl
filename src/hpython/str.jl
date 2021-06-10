"""
    pystr_decodeutf8(py, x)

Convert `x` to a Python string, assuming that `pointer(x)` points to a UTF8 encoded string of length `sizeof(x)`.

This works at least for `String`, `SubString{String}` and `Vector{UInt8}`.
"""
pystr_decodeutf8(py::Context, x::Ptr, len::Integer) = py.newhdl(py._c.PyUnicode_DecodeUTF8(x, len, C_NULL))
pystr_decodeutf8(py::Context, x) = pystr_decodeutf8(py, pointer(x), sizeof(x))
(f::Builtin{:str_decodeutf8})(args...) = pystr_decodeutf8(f.ctx, args...)

"""
    pystr_encodeutf8(py, x)

Encode the Python string `x` as UTF8 `bytes`.
"""
function pystr_encodeutf8(py::Context, x::PyAnyHdl)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyUnicode_AsUTF8String(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
(f::Builtin{:str_encodeutf8})(args...) = pystr_encodeutf8(f.ctx, args...)

"""
    pystr([T], py, x)

Convert `x` to a Python string. Optionally convert that to a `T`.
"""
pystr(py::Context, x::Union{String,SubString{String}}) = pystr_decodeutf8(py, x)
pystr(py::Context, x::AbstractString) = pystr(convert(String, x))
function pystr(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyObject_Str(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
pystr(::Type{T}, py::Context, x) where {T} = pystr_convert(T, py, pystr(py, x).auto)
(f::Builtin{:str})(x) = pystr(f.ctx, x)
(f::Builtin{:str})(::Type{T}, x) where {T} = pystr(T, f.ctx, x)

"""
    pystr_convert(T, py, x)

Convert Python string `x` to a `T`.

Allowed types are:
- `String`: On error, returns an empty string.
- `Vector{UInt8}` or `Vector`: This does a no-copy conversion, so is only valid while the bytes object is alive. On error, returns an empty array.
- `Symbol`: On error, returns `Symbol()`.
- `Char`: On error, returns `Char(0) = '\\0'`.
"""
function pystr_convert(::Type{T}, py::Context, x::PyAnyHdl) where {T}
    py.bytes_convert(T, py.str_encodeutf8(x).auto)
end
(f::Builtin{:str_convert})(::Type{T}, x) where {T} = pystr_convert(T, f.ctx, x)
