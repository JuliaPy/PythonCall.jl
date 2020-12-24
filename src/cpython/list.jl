@cdef :PyList_New PyPtr (Py_ssize_t,)
@cdef :PyList_Append Cint (PyPtr, PyPtr)
@cdef :PyList_AsTuple PyPtr (PyPtr,)

const PyList_Type__ref = Ref(PyPtr())
PyList_Type() = pyglobal(PyList_Type__ref, :PyList_Type)

PyList_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_LIST_SUBCLASS)

PyList_CheckExact(o) = Py_TypeCheckExact(o, PyList_Type())

PyList_From(xs::Union{Tuple,AbstractVector}) = PyList_FromIter(xs)

PyList_FromIter(xs) = begin
    r = PyList_New(0)
    isnull(r) && return PyPtr()
    try
        for (i,x) in enumerate(xs)
            xo = PyObject_From(x)
            isnull(xo) && (Py_DecRef(r); return PyPtr())
            err = PyList_Append(r, xo)
            Py_DecRef(xo)
            ism1(err) && (Py_DecRef(r); return PyPtr())
        end
        return r
    catch err
        Py_DecRef(r)
        PyErr_SetString(PyExc_Exception(), "Julia error: $err")
        return PyPtr()
    end
end
