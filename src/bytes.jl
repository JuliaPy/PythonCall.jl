const pybytestype = PyLazyObject(() -> pybuiltins.bytes)
export pybytestype

pyisbytes(o::AbstractPyObject) = pytypecheckfast(o, C.Py_TPFLAGS_BYTES_SUBCLASS)
export pyisbytes

pybytes(args...; opts...) = pybytestype(args...; opts...)
pybytes(x::Vector{UInt8}) = check(C.PyBytes_FromStringAndSize(pointer(x), length(x)))
export pybytes

pybytes_asjuliastring(o::AbstractPyObject) = GC.@preserve o begin
    buf = Ref{Ptr{Cchar}}()
    len = Ref{C.Py_ssize_t}()
    check(C.PyBytes_AsStringAndSize(o, buf, len))
    unsafe_string(buf[], len[])
end
