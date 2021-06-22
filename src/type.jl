pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(getptr(x_))))
export pytype
