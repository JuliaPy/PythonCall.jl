PyUnicode_DecodeUTF8(x, n, e) = ccall(POINTERS.PyUnicode_DecodeUTF8, PyPtr, (Ptr{Cchar}, Py_ssize_t, Ptr{Cvoid}), x, n, e)
PyUnicode_AsUTF8String(o) = ccall(POINTERS.PyUnicode_AsUTF8String, PyPtr, (PyPtr,), o)
PyUnicode_InternInPlace(o) = ccall(POINTERS.PyUnicode_InternInPlace, Cvoid, (Ptr{PyPtr},), o)

PyUnicode_Type() = POINTERS.PyUnicode_Type

PyUnicode_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_UNICODE_SUBCLASS)
PyUnicode_CheckExact(o) = Py_TypeCheckExact(o, PyUnicode_Type())

PyUnicode_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyUnicode_DecodeUTF8(pointer(s), sizeof(s), C_NULL)

PyUnicode_AsString(o) = begin
    b = PyUnicode_AsUTF8String(o)
    isnull(b) && return ""
    r = PyBytes_AsString(b)
    Py_DecRef(b)
    r
end

PyUnicode_AsVector(o, ::Type{T} = UInt8) where {T} = begin
    b = PyUnicode_AsUTF8String(o)
    isnull(b) && return T[]
    r = PyBytes_AsVector(b, T)
    Py_DecRef(b)
    r
end

PyUnicode_TryConvertRule_string(o, ::Type{String}) = begin
    r = PyUnicode_AsString(o)
    isempty(r) && PyErr_IsSet() && return -1
    putresult(r)
end

PyUnicode_TryConvertRule_vector(o, ::Type{Vector{X}}) where {X} = begin
    r = PyUnicode_AsVector(o, X)
    isempty(r) && PyErr_IsSet() && return -1
    putresult(r)
end

PyUnicode_TryConvertRule_symbol(o, ::Type{Symbol}) = begin
    r = PyUnicode_AsString(o)
    isempty(r) && PyErr_IsSet() && return -1
    putresult(Symbol(r))
end

PyUnicode_TryConvertRule_char(o, ::Type{Char}) = begin
    r = PyUnicode_AsString(o)
    isempty(r) && PyErr_IsSet() && return -1
    length(r) == 1 || return 0
    putresult(first(r))
end
