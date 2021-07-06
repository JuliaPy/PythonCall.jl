pyfraction(x::Rational) = pyfraction(numerator(x), denominator(x))
pyfraction(x, y) = pyfractiontype(x, y)
pyfraction(x) = pyfractiontype(x)
pyfraction() = pyfractiontype()
export pyfraction

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{R}, x::Py, ::Type{Rational{T0}}=Utils._type_lb(R), ::Type{Rational{T1}}=Utils._type_ub(R)) where {R<:Rational,T0,T1}
    # numerator
    a_ = x.numerator
    r = pytryconvert(Utils._typeintersect(Integer,T1), a_)
    pydel!(a_)
    pyconvert_isunconverted(r) && return r
    a = pyconvert_result(T1, r)
    # numerator
    b_ = x.denominator
    r = pytryconvert(Utils._typeintersect(Integer,T1), b_)
    pydel!(b_)
    pyconvert_isunconverted(r) && return r
    b = pyconvert_result(T1, r)
    # success
    a, b = promote(a, b)
    pyconvert_return(Rational{Union{T0,typeof(a),typeof(b)}}(a, b))
end

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{T}, x::Py) where {T<:Number}
    r = pyconvert_rule_fraction(Rational{<:Integer}, x)
    pyconvert_isunconverted(r) && return r
    pyconvert_tryconvert(T, pyconvert_result(Rational{<:Integer}, r))
end
