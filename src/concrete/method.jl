# :PyInstanceMethod_New => (PyPtr,) => PyPtr,

"""
    pymethod(x)

Convert callable `x` to a Python instance method.
"""
pymethod(x) = pynew(errcheck(@autopy x C.PyInstanceMethod_New(getptr(x_))))
export pymethod
