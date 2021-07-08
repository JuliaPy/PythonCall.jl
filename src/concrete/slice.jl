# :PySlice_New => (PyPtr, PyPtr, PyPtr) => PyPtr,

pyslice(x, y, z=pybuiltins.None) = pynew(errcheck(@autopy x y z C.PySlice_New(getptr(x_), getptr(y_), getptr(z_))))
pyslice(y=pybuiltins.None) = pyslice(pybuiltins.None, y, pybuiltins.None)
export pyslice

pyisslice(x) = pytypecheck(x, pybuiltins.slice)
