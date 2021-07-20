"""
    pytype(x)

The Python `type` of `x`.
"""
pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(getptr(x_))))
pytype(name, bases, dict) = pybuiltins.type(name, ispy(bases) && pyistype(bases) ? pytuple((bases,)) : pytuple(bases), pydict(dict))
export pytype

"""
    pyclass(name, bases=(); members...)

Construct a new Python type with the given name, bases and members.

Equivalent to `pytype(name, bases, members)`.
"""
pyclass(name, bases=(); dict...) = pytype(name, bases, pystrdict_fromiter(dict))
export pyclass

pyistype(x) = pytypecheckfast(x, C.Py_TPFLAGS_TYPE_SUBCLASS)

pytypecheck(x, t) = (@autopy x t C.Py_TypeCheck(getptr(x_), getptr(t_))) == 1
pytypecheckfast(x, f) = (@autopy x C.Py_TypeCheckFast(getptr(x_), f)) == 1
