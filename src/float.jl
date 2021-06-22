# :PyFloat_FromDouble => (Cdouble,) => PyPtr,
# :PyFloat_AsDouble => (PyPtr,) => Cdouble,

pyfloat(x::Real=0.0) = pynew(errcheck(C.PyFloat_FromDouble(x)))
pyfloat(x) = ispy(x) ? pynew(errcheck(C.PyNumber_Float(getptr(x)))) : pyfloat(convert(Real, x))
export pyfloat

pyfloat_asdouble(x) = errcheck_ambig(@autopy x C.PyFloat_AsDouble(getptr(x_)))

function pyconvert_rule_float(::Type{T}, x::Py) where {T<:Number}
    val = pyfloat_asdouble(x)
    if T in (Float16, Float32, Float64, BigFloat)
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end
