PyType_CheckBuffer(t) = begin
    p = UnsafePtr{PyTypeObject}(t).as_buffer[]
    !isnull(p) && !isnull(p.get[!])
end

PyObject_CheckBuffer(o) = PyType_CheckBuffer(Py_Type(o))

PyObject_GetBuffer(o, b, flags) = begin
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if isnull(p) || isnull(p.get[])
        PyErr_SetString(unsafe_load(Ptr{PyPtr}(pyglobal(:PyExc_TypeError))), "a bytes-like object is required, not '$(String(UnsafePtr{PyTypeObject}(Py_Type(o)).name[]))'")
        return Cint(-1)
    end
    ccall(p.get[!], Cint, (PyPtr, Ptr{Py_buffer}, Cint), o, b, flags)
end

PyBuffer_Release(_b) = begin
    b = UnsafePtr(Base.unsafe_convert(Ptr{Py_buffer}, _b))
    o = b.obj[]
    isnull(o) && return
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if (!isnull(p) && !isnull(p.release[]))
        ccall(p.release[!], Cvoid, (PyPtr, Ptr{Py_buffer}), o, b)
    end
    b.obj[] = C_NULL
    Py_DecRef(o)
    return
end
