# :PyComplex_FromDoubles => (Cdouble, Cdouble) => PyPtr,
# :PyComplex_RealAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_ImagAsDouble => (PyPtr,) => Cdouble,
# :PyComplex_AsCComplex => (PyPtr,) => Py_complex,

function pyconvert_rule_complex(::Type{T}, x::Py) where {T<:Number}
    val = pycomplex_ascomplex(x)
    if T in (Complex{Float64}, Complex{Float32}, Complex{Float16}, Complex{BigFloat})
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end
