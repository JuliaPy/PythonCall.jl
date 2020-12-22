@cdef :PyBytes_FromStringAndSize PyPtr (Ptr{Cchar}, Py_ssize_t)
@cdef :PyBytes_AsStringAndSize Cint (PyPtr, Ptr{Ptr{Cchar}}, Ptr{Py_ssize_t})

const PyBytes_Type__ref = Ref(PyPtr())
PyBytes_Type() = pyglobal(PyBytes_Type__ref, :PyBytes_Type)

PyBytes_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_BYTES_SUBCLASS)
PyBytes_CheckExact(o) = Py_TypeCheckExact(o, PyBytes_Type())

PyBytes_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyBytes_FromStringAndSize(pointer(s), sizeof(s))

PyBytes_AsString(o) = begin
    ptr = Ref{Ptr{Cchar}}()
    len = Ref{Py_ssize_t}()
    err = PyBytes_AsStringAndSize(o, ptr, len)
    ism1(err) && return ""
    Base.unsafe_string(ptr[], len[])
end

PyBytes_AsVector(o, ::Type{T}=UInt8) where {T} = begin
    T in (Int8, UInt8) || throw(MethodError(PyBytes_AsVector, (o, T)))
    ptr = Ref{Ptr{Cchar}}()
    len = Ref{Py_ssize_t}()
    err = PyBytes_AsStringAndSize(o, ptr, len)
    ism1(err) && return T[]
    copy(Base.unsafe_wrap(Vector{T}, Ptr{T}(ptr[]), len[]))
end

PyBytes_TryConvertRule_vector(o, ::Type{Vector{X}}) where {X} = begin
    v = PyBytes_AsVector(o, X)
    isempty(v) && PyErr_IsSet() && return -1
    return putresult(v)
end

PyBytes_TryConvertRule_string(o, ::Type{String}) = begin
    v = PyBytes_AsString(o)
    isempty(v) && PyErr_IsSet() && return -1
    return putresult(v)
end
