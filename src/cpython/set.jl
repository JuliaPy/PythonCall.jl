PySet_New(o) = ccall(POINTERS.PySet_New, PyPtr, (PyPtr,), o)
PyFrozenSet_New(o) = ccall(POINTERS.PyFrozenSet_New, PyPtr, (PyPtr,), o)
PySet_Add(o, v) = ccall(POINTERS.PySet_Add, Cint, (PyPtr, PyPtr), o, v)

PySet_Type() = POINTERS.PySet_Type

PySet_Check(o) = Py_TypeCheckFast(o, PySet_Type())

PySet_CheckExact(o) = Py_TypeCheckExact(o, PySet_Type())

PyFrozenSet_Type() = POINTERS.PyFrozenSet_Type

PyFrozenSet_Check(o) = Py_TypeCheckFast(o, PyFrozenSet_Type())

PyFrozenSet_CheckExact(o) = Py_TypeCheckExact(o, PyFrozenSet_Type())

PyAnySet_Check(o) = PySet_Check(o) || PyFrozenSet_Check(o)

PyAnySet_CheckExact(o) = PySet_CheckExact(o) || PyFrozenSet_CheckExact(o)

_PySet_FromIter(r::PyPtr, xs) = begin
    try
        for x in xs
            xo = PyObject_From(x)
            isnull(xo) && (Py_DecRef(r); return PyNULL)
            err = PySet_Add(r, xo)
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

PySet_FromIter(xs) = begin
    r = PySet_New(C_NULL)
    isnull(r) && return PyNULL
    _PySet_FromIter(r, xs)
end

PyFrozenSet_FromIter(xs) = begin
    r = PyFrozenSet_New(C_NULL)
    isnull(r) && return PyNULL
    _PySet_FromIter(r, xs)
end

PySet_From(x::Union{AbstractSet,AbstractVector,Tuple}) = PySet_FromIter(x)

PyFrozenSet_From(x::Union{AbstractSet,AbstractVector,Tuple}) = PyFrozenSet_FromIter(x)
