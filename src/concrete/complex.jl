# :PyComplex_FromDoubles => (Cdouble, Cdouble) => PyPtr,
# :PyComplex_RealAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_ImagAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_AsCComplex => (PyPtr,) => Py_complex,

"""
    pycomplex(x=0.0)
    pycomplex(re, im)

Convert `x` to a Python `complex`, or create one from given real and imaginary parts.
"""
pycomplex(x::Real=0.0, y::Real=0.0) = pynew(errcheck(C.PyComplex_FromDoubles(x, y)))
pycomplex(x::Complex) = pycomplex(real(x), imag(x))
pycomplex(x) = pybuiltins.complex(x)
pycomplex(x, y) = pybuiltins.complex(x, y)
export pycomplex

pyiscomplex(x) = pytypecheck(x, pybuiltins.complex)

function pycomplex_ascomplex(x)
    c = @autopy x C.PyComplex_AsCComplex(getptr(x_))
    c.real == -1 && c.imag == 0 && errcheck()
    return Complex(c.real, c.imag)
end

function pyconvert_rule_complex(::Type{T}, x::Py) where {T<:Number}
    val = pycomplex_ascomplex(x)
    if T in (Complex{Float64}, Complex{Float32}, Complex{Float16}, Complex{BigFloat})
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end

pyconvert_rule_fast(::Type{Complex{Float64}}, x::Py) =
    if pyiscomplex(x)
        pyconvert_return(pycomplex_ascomplex(x))
    else
        pyconvert_unconverted()
    end
