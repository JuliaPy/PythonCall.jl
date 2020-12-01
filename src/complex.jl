const pycomplextype = pylazyobject(() -> pybuiltins.complex)
export pycomplextype

pycomplex(args...; opts...) = pycomplextype(args...; opts...)
pycomplex(x::Complex) = pycomplex(real(x), imag(x))
export pycomplex

pyiscomplex(o::PyObject) = pytypecheck(o, pycomplextype)
export pyiscomplex

function pycomplex_tryconvert(::Type{T}, o::PyObject) where {T}
    x = check(C.PyComplex_RealAsDouble(o), true)
    y = check(C.PyComplex_ImagAsDouble(o), true)
    z = Complex(x, y)
    if (S = _typeintersect(T, Complex{Cdouble})) != Union{}
        convert(S, z)
    elseif (S = _typeintersect(T, Complex)) != Union{}
        tryconvert(S, z)
    elseif (S = _typeintersect(T, Real)) != Union{}
        tryconvert(S, z)
    else
        tryconvert(T, z)
    end
end
