# :PyInstanceMethod_New => (PyPtr,) => PyPtr,

pymethod(x) = pynew(errcheck(@autopy x C.PyInstanceMethod_New(getptr(x_))))
export pymethod
