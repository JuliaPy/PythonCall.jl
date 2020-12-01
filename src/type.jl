const pytypetype = pylazyobject(() -> pybuiltins.type)
export pytypetype

pyistype(o::PyObject) = pytypecheckfast(o, C.Py_TPFLAGS_TYPE_SUBCLASS)
export pyistype

pytype(o::PyObject) = pyborrowedobject(C.Py_Type(o))
export pytype

pytypecheck(o::PyObject, t::PyObject) = !iszero(C.Py_TypeCheck(o, t))

pytypecheckfast(o::PyObject, f::Integer) = !iszero(C.Py_TypeCheckFast(o, f))
