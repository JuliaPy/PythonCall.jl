@cdef :PyTuple_New PyPtr (Py_ssize_t,)
@cdef :PyTuple_Size Py_ssize_t (PyPtr,)
@cdef :PyTuple_GetItem PyPtr (PyPtr, Py_ssize_t)
@cdef :PyTuple_SetItem Cint (PyPtr, Py_ssize_t, PyPtr)

function PyTuple_From(xs::Tuple)
    t = PyTuple_New(length(xs))
    isnull(t) && return PyPtr()
    for (i,x) in enumerate(xs)
        xo = PyObject_From(x)
        isnull(xo) && (Py_DecRef(t); return PyPtr())
        err = PyTuple_SetItem(t, i-1, xo) # steals xo
        ism1(err) && (Py_DecRef(t); return PyPtr())
    end
    return t
end
