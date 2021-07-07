pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(getptr(x_))))
pytype(name, bases, dict) = pybuiltins.type(name, pytuple(bases), pydict(dict))
export pytype

pyistype(x) = pytypecheckfast(x, C.Py_TPFLAGS_TYPE_SUBCLASS)

pytypecheck(x, t) = (@autopy x t C.Py_TypeCheck(getptr(x_), getptr(t_))) == 1
pytypecheckfast(x, f) = (@autopy x C.Py_TypeCheckFast(getptr(x_), f)) == 1
