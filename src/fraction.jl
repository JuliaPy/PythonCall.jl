const pyfractiontype = PyLazyObject(() -> pyimport("fractions").Fraction)
export pyfractiontype

pyfraction(args...; opts...) = pyfractiontype(args...; opts...)
pyfraction(x::Rational) = pyfraction(numerator(x), denominator(x))
export pyfraction

pyisfraction(o::AbstractPyObject) = pytypecheck(o, pyfractiontype)
export pyisfraction

function pyfraction_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:Rational}
    tryconvert(T, pyfraction_tryconvert(Rational{BigInt}, o))
end

function pyfraction_tryconvert(::Type{Rational{T}}, o::AbstractPyObject) where {T<:Integer}
    x = pyint_tryconvert(T, o.numerator)
    x === PyConvertFail() && return x
    y = pyint_tryconvert(T, o.denominator)
    y === PyConvertFail() && return y
    Rational{T}(x, y)
end

function pyfraction_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if (S = _typeintersect(T, Rational)) != Union{}
        pyfraction_tryconvert(S, o)
    elseif (S = _typeintersect(T, Integer)) != Union{}
        o.denominator == pyint(1) ? pyint_tryconvert(S, o.numerator) : PyConvertFail()
    elseif (S = _typeintersect(T, Number)) != Union{}
        tryconvert(S, pyfraction_tryconvert(Rational{BigInt}, o))
    else
        tryconvert(T, pyfraction_tryconvert(Rational{BigInt}, o))
    end
end
