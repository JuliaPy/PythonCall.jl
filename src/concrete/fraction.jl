pyfraction(x::Rational) = pyfraction(numerator(x), denominator(x))
pyfraction(x, y) = pyfractiontype(x, y)
pyfraction(x) = pyfractiontype(x)
pyfraction() = pyfractiontype()
export pyfraction

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{R}, x::Py, ::Type{Rational{T0}}=Utils._type_lb(R), ::Type{Rational{T1}}=Utils._type_ub(R)) where {R<:Rational,T0,T1}
    a = @pyconvert_and_del(Utils._typeintersect(Integer, T1), x.numerator)
    b = @pyconvert_and_del(Utils._typeintersect(Integer, T2), x.denominator)
    a, b = promote(a, b)
    T2 = Utils._promote_type_bounded(T0, typeof(a), typeof(b), T1)
    pyconvert_return(Rational{T2}(a, b))
end

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{T}, x::Py) where {T<:Number}
    pyconvert_tryconvert(T, @pyconvert(Rational{<:Integer}, x))
end
