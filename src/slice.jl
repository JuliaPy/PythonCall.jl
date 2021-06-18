# :PySlice_New => (PyPtr, PyPtr, PyPtr) => PyPtr,

pyslice(x, y, z=pyNone) = setptr!(pynew(), errcheck(@autopy x y z C.PySlice_New(getptr(x_), getptr(y_), getptr(z_))))
pyslice(y=pyNone) = pyslice(pyNone, y, pyNone)
export pyslice
