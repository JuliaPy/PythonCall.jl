"""
    pyiter(x)

Equivalent to `iter(x)` in Python.
"""
pyiter(x) = pynew(errcheck(@autopy x C.PyObject_GetIter(getptr(x_))))
export pyiter

"""
    pynext(x)

Equivalent to `next(x)` in Python.
"""
pynext(x) = pybuiltins.next(x)
export pynext

"""
    unsafe_pynext(x)

Return the next item in the iterator `x`. When there are no more items, return NULL.
"""
unsafe_pynext(x::Py) = pynew(errcheck_ambig(C.PyIter_Next(getptr(x))))
