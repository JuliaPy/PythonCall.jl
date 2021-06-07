PyTuple_New(n) = ccall(POINTERS.PyTuple_New, PyPtr, (Py_ssize_t,), n)
PyTuple_Size(o) = ccall(POINTERS.PyTuple_Size, Py_ssize_t, (PyPtr,), o)
PyTuple_GetItem(o, i) = ccall(POINTERS.PyTuple_GetItem, PyPtr, (PyPtr, Py_ssize_t), o, i)
PyTuple_SetItem(o, i, v) = ccall(POINTERS.PyTuple_SetItem, Cint, (PyPtr, Py_ssize_t, PyPtr), o, i, v)

PyTuple_Type() = POINTERS.PyTuple_Type

PyTuple_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_TUPLE_SUBCLASS)

PyTuple_CheckExact(o) = Py_TypeCheckExact(o, PyTuple_Type())

PyTuple_From(x::Union{Tuple,AbstractVector}) = PyTuple_FromIter(x)

PyTuple_FromIter(xs::Tuple) = begin
    t = PyTuple_New(length(xs))
    isnull(t) && return PyNULL
    for (i, x) in enumerate(xs)
        xo = PyObject_From(x)
        isnull(xo) && (Py_DecRef(t); return PyNULL)
        err = PyTuple_SetItem(t, i - 1, xo) # steals xo
        ism1(err) && (Py_DecRef(t); return PyNULL)
    end
    return t
end

PyTuple_FromIter(xs) = begin
    y = PyList_FromIter(xs)
    isnull(y) && return PyNULL
    t = PyList_AsTuple(y)
    Py_DecRef(y)
    return t
end
