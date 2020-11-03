const pybytestype = PyLazyObject(() -> pybuiltins.bytes)
export pybytestype

pyisbytes(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_BYTES_SUBCLASS)
export pyisbytes

pybytes(args...; opts...) = pybytestype(args...; opts...)
export pybytes

function pybytes_asstringandsize(o::AbstractPyObject)
    buf = Ref{Ptr{Cchar}}()
    len = Ref{CPy_ssize_t}()
    err = cpycall_int(Val(:PyBytes_AsStringAndSize), o, buf, len)
    (buf[], len[])
end

pybytes_asjuliastring(o::AbstractPyObject) = GC.@preserve o unsafe_string(pybytes_asstringandsize(o)...)
