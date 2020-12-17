@cdef :PyUnicode_DecodeUTF8 PyPtr (Ptr{Cchar}, Py_ssize_t, Ptr{Cvoid})
@cdef :PyUnicode_AsUTF8String PyPtr (PyPtr,)
@cdef :PyUnicode_InternInPlace Cvoid (Ptr{PyPtr},)

const PyUnicode_Type__ref = Ref(PyPtr())
PyUnicode_Type() = pyglobal(PyUnicode_Type__ref, :PyUnicode_Type)

PyUnicode_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_UNICODE_SUBCLASS)
PyUnicode_CheckExact(o) = Py_TypeCheckExact(o, PyUnicode_Type())

PyUnicode_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyUnicode_DecodeUTF8(pointer(s), sizeof(s), C_NULL)

PyUnicode_TryConvertRule_string(o, ::Type{T}, ::Type{String}) where {T} = begin
    b = PyUnicode_AsUTF8String(o)
    isnull(b) && return -1
    r = PyBytes_TryConvertRule_string(b, String, String)
    Py_DecRef(b)
    r == 1 || return r
    moveresult(String, T)
end

PyUnicode_TryConvertRule_vector(o, ::Type{T}, ::Type{S}) where {T, S<:Vector} = begin
    b = PyUnicode_AsUTF8String(o)
    isnull(b) && return -1
    r = PyBytes_TryConvertRule_vector(b, S, S)
    Py_DecRef(b)
    r == 1 || return r
    moveresult(S, T)
end

PyUnicode_TryConvertRule_symbol(o, ::Type{T}, ::Type{Symbol}) where {T} = begin
    r = PyUnicode_TryConvertRule_string(o, String, String)
    r == 1 || return r
    putresult(T, Symbol(takeresult(String)))
end

PyUnicode_TryConvertRule_char(o, ::Type{T}, ::Type{Char}) where {T} = begin
    r = PyUnicode_TryConvertRule_string(o, String, String)
    r == 1 || return r
    s = takeresult(String)
    length(s) == 1 || return 0
    putresult(T, first(s))
end
