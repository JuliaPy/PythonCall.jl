const pycomplextype = PyLazyObject(() -> pybuiltins.complex)
export pycomplextype

pycomplex(args...; opts...) = pycomplextype(args...; opts...)
pycomplex(x::Complex) = pycomplex(real(x), imag(x))
export pycomplex

pyiscomplex(o::AbstractPyObject) = pytypecheck(o, pycomplextype)
export pyiscomplex

function pycomplex_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    x = cpycall_num_ambig(Val(:PyComplex_RealAsDouble), Cdouble, o)
    y = cpycall_num_ambig(Val(:PyComplex_ImagAsDouble), Cdouble, o)
    z = Complex(x, y)
    z == Complex(-1.0, 0.0) && pyerrcheck()
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
