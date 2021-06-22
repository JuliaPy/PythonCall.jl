pyiter(x) = pynew(errcheck(@autopy x C.PyObject_GetIter(getptr(x_))))
export pyiter

pynext(x::Py) = pynew(errcheck_nullable(C.PyIter_Next(getptr(x))))
export pyiter
