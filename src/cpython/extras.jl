Py_Type(x::PyPtr) = PyPtr(UnsafePtr(x).type[!])

PyObject_Type(x::PyPtr) = (t=Py_Type(x); Py_IncRef(t); t)
