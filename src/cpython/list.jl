PyList_New(n) = ccall(POINTERS.PyList_New, PyPtr, (Py_ssize_t,), n)
PyList_Append(o, v) = ccall(POINTERS.PyList_Append, Cint, (PyPtr, PyPtr), o, v)
PyList_AsTuple(o) = ccall(POINTERS.PyList_AsTuple, PyPtr, (PyPtr,), o)

PyList_Type() = POINTERS.PyList_Type

PyList_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_LIST_SUBCLASS)

PyList_CheckExact(o) = Py_TypeCheckExact(o, PyList_Type())

PyList_From(xs::Union{Tuple,AbstractVector}) = PyList_FromIter(xs)

PyList_FromIter(xs) = begin
    r = PyList_New(0)
    isnull(r) && return PyNULL
    try
        for (i, x) in enumerate(xs)
            xo = PyObject_From(x)
            isnull(xo) && (Py_DecRef(r); return PyNULL)
            err = PyList_Append(r, xo)
            Py_DecRef(xo)
            ism1(err) && (Py_DecRef(r); return PyNULL)
        end
        return r
    catch err
        Py_DecRef(r)
        PyErr_SetString(PyExc_Exception(), "Julia error: $err")
        return PyNULL
    end
end
