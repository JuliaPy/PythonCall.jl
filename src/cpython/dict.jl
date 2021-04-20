PyDict_New() = ccall(POINTERS.PyDict_New, PyPtr, ())
PyDict_GetItem(o, k) = ccall(POINTERS.PyDict_GetItem, PyPtr, (PyPtr, PyPtr), o, k)
PyDict_GetItemString(o, k) = ccall(POINTERS.PyDict_GetItemString, PyPtr, (PyPtr, Cstring), o, k)
PyDict_SetItem(o, k, v) = ccall(POINTERS.PyDict_SetItem, Cint, (PyPtr, PyPtr, PyPtr), o, k, v)
PyDict_SetItemString(o, k, v) = ccall(POINTERS.PyDict_SetItemString, Cint, (PyPtr, Cstring, PyPtr), o, k, v)
PyDict_DelItemString(o, k) = ccall(POINTERS.PyDict_DelItemString, Cint, (PyPtr, Cstring), o, k)

PyDict_Type() = POINTERS.PyDict_Type

PyDict_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_DICT_SUBCLASS)

PyDict_CheckExact(o) = Py_TypeCheckExact(o, PyDict_Type())

PyDict_FromPairs(kvs) = begin
    r = PyDict_New()
    isnull(r) && return PyNULL
    try
        for (k, v) in kvs
            ko = PyObject_From(k)
            isnull(ko) && (Py_DecRef(r); return PyNULL)
            vo = PyObject_From(v)
            isnull(vo) && (Py_DecRef(r); Py_DecRef(ko); return PyNULL)
            err = PyDict_SetItem(r, ko, vo)
            Py_DecRef(ko)
            Py_DecRef(vo)
            ism1(err) && (Py_DecRef(r); return PyNULL)
        end
        return r
    catch err
        Py_DecRef(r)
        PyErr_SetString(PyExc_Exception(), "Julia error: $err")
        return PyNULL
    end
end

PyDict_FromStringPairs(kvs) = begin
    r = PyDict_New()
    isnull(r) && return PyNULL
    try
        for (k, v) in kvs
            vo = PyObject_From(v)
            isnull(vo) && (Py_DecRef(r); return PyNULL)
            err = PyDict_SetItemString(r, string(k), vo)
            Py_DecRef(vo)
            ism1(err) && (Py_DecRef(r); return PyNULL)
        end
        return r
    catch err
        Py_DecRef(r)
        PyErr_SetString(PyExc_Exception(), "Julia error: $err")
        return PyNULL
    end
end

PyDict_From(x::AbstractDict) = PyDict_FromPairs(x)
PyDict_From(x::AbstractDict{String}) = PyDict_FromStringPairs(x)
PyDict_From(x::AbstractDict{Symbol}) = PyDict_FromStringPairs(x)
PyDict_From(x::NamedTuple) = PyDict_FromStringPairs(pairs(x))
PyDict_From(x::Base.Iterators.Pairs) = PyDict_FromPairs(x)
PyDict_From(x::Base.Iterators.Pairs{String}) = PyDict_FromStringPairs(x)
PyDict_From(x::Base.Iterators.Pairs{Symbol}) = PyDict_FromStringPairs(x)
