Py_Type(x::PyPtr) = PyPtr(UnsafePtr(x).type[!])

PyObject_Type(x::PyPtr) = (t=Py_Type(x); Py_IncRef(t); t)

Py_TypeCheck(o::PyPtr, t::PyPtr) = PyType_IsSubtype(Py_Type(o), t)
Py_TypeCheckFast(o::PyPtr, f::Integer) = PyType_IsSubtypeFast(Py_Type(o), f)

PyType_IsSubtypeFast(t::PyPtr, f::Integer) = Cint(!iszero(UnsafePtr{PyTypeObject}(t).flags[] & f))
