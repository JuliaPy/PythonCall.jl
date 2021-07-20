"""
    pyiter(x)

Equivalent to `iter(x)` in Python.
"""
pyiter(x) = pynew(errcheck(@autopy x C.PyObject_GetIter(getptr(x_))))
export pyiter

pynext(x::Py) = pynew(errcheck_ambig(C.PyIter_Next(getptr(x))))
export pyiter
