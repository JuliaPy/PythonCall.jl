# :PyComplex_FromDoubles => (Cdouble, Cdouble) => PyPtr,
# :PyComplex_RealAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_ImagAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_AsCComplex => (PyPtr,) => Py_complex,

pycomplex(x::Real=0.0, y::Real=0.0) = setptr!(pynew(), errcheck(C.PyComplex_FromDoubles(x, y)))
pycomplex(x::Complex) = pycomplex(real(x), imag(x))
pycomplex(x) = ispy(x) ? pycomplextype(x) : pycomplex(convert(Complex{Cdouble}, x))
pycomplex(x, y) = (ispy(x) || ispy(y)) ? pycomplextype(x, y) : pycomplex(convert(Cdouble, x), convert(Cdouble, y))
export pycomplex
