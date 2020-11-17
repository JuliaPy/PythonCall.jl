const pytypetype = PyLazyObject(() -> pybuiltins.type)
export pytypetype

pyistype(o::AbstractPyObject) = pytypecheckfast(o, C.Py_TPFLAGS_TYPE_SUBCLASS)
export pyistype

pytype(o::AbstractPyObject) = pynewobject(C.Py_Type(o), true)
export pytype

pytypecheck(o::AbstractPyObject, t::AbstractPyObject) = !iszero(C.Py_TypeCheck(o, t))

pytypecheckfast(o::AbstractPyObject, f::Integer) = !iszero(C.Py_TypeCheckFast(o, f))
