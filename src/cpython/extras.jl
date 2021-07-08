Py_Type(x::PyPtr) = PyPtr(UnsafePtr(x).type[!])

PyObject_Type(x::PyPtr) = (t=Py_Type(x); Py_IncRef(t); t)

Py_TypeCheck(o::PyPtr, t::PyPtr) = PyType_IsSubtype(Py_Type(o), t)
Py_TypeCheckFast(o::PyPtr, f::Integer) = PyType_IsSubtypeFast(Py_Type(o), f)

PyType_IsSubtypeFast(t::PyPtr, f::Integer) = Cint(!iszero(UnsafePtr{PyTypeObject}(t).flags[] & f))

PyMemoryView_GET_BUFFER(m::PyPtr) = Ptr{Py_buffer}(UnsafePtr{PyMemoryViewObject}(m).view)

PyType_CheckBuffer(t::PyPtr) = begin
    p = UnsafePtr{PyTypeObject}(t).as_buffer[]
    return p != C_NULL && p.get[!] != C_NULL
end

PyObject_CheckBuffer(o::PyPtr) = PyType_CheckBuffer(Py_Type(o))

PyObject_GetBuffer(o::PyPtr, b, flags) = begin
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if p == C_NULL || p.get[!] == C_NULL
        PyErr_SetString(
            POINTERS.PyExc_TypeError,
            "a bytes-like object is required, not '$(String(UnsafePtr{PyTypeObject}(Py_Type(o)).name[]))'",
        )
        return Cint(-1)
    end
    return ccall(p.get[!], Cint, (PyPtr, Ptr{Py_buffer}, Cint), o, b, flags)
end

PyBuffer_Release(_b) = begin
    b = UnsafePtr(Base.unsafe_convert(Ptr{Py_buffer}, _b))
    o = b.obj[]
    o == C_NULL && return
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if (p != C_NULL && p.release[!] != C_NULL)
        ccall(p.release[!], Cvoid, (PyPtr, Ptr{Py_buffer}), o, b)
    end
    b.obj[] = C_NULL
    Py_DecRef(o)
    return
end
