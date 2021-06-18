# :PyFloat_FromDouble => (Cdouble,) => PyPtr,
# :PyFloat_AsDouble => (PyPtr,) => Cdouble,

pyfloat(x::Real=0.0) = setptr!(pynew(), errcheck(C.PyFloat_FromDouble(x)))
pyfloat(x) = ispy(x) ? setptr!(pynew(), errcheck(C.PyNumber_Float(getptr(x)))) : pyfloat(convert(Real, x))
export pyfloat
