const pytypetype = PyLazyObject(() -> pybuiltins.type)
export pytypetype

pyistype(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_TYPE_SUBCLASS)
export pyistype

pytype(o::AbstractPyObject) = pynewobject(cpytype(pyptr(o)), true)
export pytype

pytypecheck(o::AbstractPyObject, t::AbstractPyObject) = cpytypecheck(pyptr(o), pyptr(t))

pytypecheckfast(o::AbstractPyObject, f::Integer) = cpytypecheckfast(pyptr(o), f)
