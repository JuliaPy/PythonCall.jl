@cdef :PyBytes_FromStringAndSize PyPtr (Ptr{Cchar}, Py_ssize_t)
@cdef :PyBytes_AsStringAndSize Cint (PyPtr, Ptr{Ptr{Cchar}}, Ptr{Py_ssize_t})

const PyBytes_Type__ref = Ref(PyPtr())
PyBytes_Type() = pyglobal(PyBytes_Type__ref, :PyBytes_Type)

PyBytes_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_BYTES_SUBCLASS)
PyBytes_CheckExact(o) = Py_TypeCheckExact(o, PyBytes_Type())

PyBytes_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyBytes_FromStringAndSize(pointer(s), sizeof(s))

PyBytes_TryConvertRule_vector(o, ::Type{T}, ::Type{Vector{X}}) where {T,X} = begin
    ptr = Ref{Ptr{Cchar}}()
    len = Ref{Py_ssize_t}()
    err = PyBytes_AsStringAndSize(o, ptr, len)
    ism1(err) && return -1
    v = copy(Base.unsafe_wrap(Vector{X}, Ptr{X}(ptr[]), len[]))
    return putresult(T, v)
end

PyBytes_TryConvertRule_string(o, ::Type{T}, ::Type{String}) where {T} = begin
    ptr = Ref{Ptr{Cchar}}()
    len = Ref{Py_ssize_t}()
    err = PyBytes_AsStringAndSize(o, ptr, len)
    ism1(err) && return -1
    v = Base.unsafe_string(ptr[], len[])
    return putresult(T, v)
end
