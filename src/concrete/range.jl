pyrange(x, y, z) = pybuiltins.range(x, y, z)
pyrange(x, y) = pybuiltins.range(x, y)
pyrange(y) = pybuiltins.range(y)
export pyrange

pyrange_fromrange(x::AbstractRange) = pyrange(first(x), last(x) + sign(step(x)), step(x))

pyisrange(x) = pytypecheck(x, pybuiltins.range)

function pyconvert_rule_range(::Type{R}, x::Py, ::Type{StepRange{T0,S0}}=Utils._type_lb(R), ::Type{StepRange{T1,S1}}=Utils._type_ub(R)) where {R<:StepRange,T0,S0,T1,S1}
    # start
    a_ = x.start
    r = pytryconvert(Utils._typeintersect(Integer, T1), a_)
    pyconvert_isunconverted(r) && return r
    a = pyconvert_result(r)
    pydel!(a_)
    # step
    b_ = x.step
    r = pytryconvert(Utils._typeintersect(Integer, S1), b_)
    pyconvert_isunconverted(r) && return r
    b = pyconvert_result(r)
    pydel!(b_)
    # stop
    c_ = x.stop
    r = pytryconvert(Utils._typeintersect(Integer, T1), c_)
    pyconvert_isunconverted(r) && return r
    c = pyconvert_result(r)
    pydel!(c_)
    # success
    a′, c′ = promote(a, c - oftype(c, sign(b)))
    pyconvert_return(StepRange{Union{T0, typeof(a′), typeof(c′)}, Union{S0, typeof(b)}}(a′, b, c′))
end

function pyconvert_rule_range(::Type{R}, x::Py, ::Type{UnitRange{T0}}=Utils._type_lb(R), ::Type{UnitRange{T1}}=Utils._type_ub(R)) where {R<:UnitRange,T0,T1}
    # start
    a_ = x.start
    r = pytryconvert(Utils._typeintersect(Integer, T1), a_)
    pyconvert_isunconverted(r) && return r
    a = pyconvert_result(r)
    pydel!(a_)
    # step
    b_ = x.step
    r = pytryconvert(Utils._typeintersect(Integer, T1), b_)
    pyconvert_isunconverted(r) && return r
    b = pyconvert_result(r)
    pydel!(b_)
    b == 1 || return pyconvert_unconverted()
    # stop
    c_ = x.stop
    r = pytryconvert(Utils._typeintersect(Integer, T1), c_)
    pyconvert_isunconverted(r) && return r
    c = pyconvert_result(r)
    pydel!(c_)
    # success
    a′, c′ = promote(a, c - oftype(c, 1))
    pyconvert_return(UnitRange{Union{T0, typeof(a′), typeof(c′)}}(a′, c′))
end

pyconvert_rule_fast(::Type{T}, x::Py) where {T<:StepRange} = pyisrange(x) ? pyconvert_rule_range(T, x) : pyconvert_unconverted()
pyconvert_rule_fast(::Type{T}, x::Py) where {T<:UnitRange} = pyisrange(x) ? pyconvert_rule_range(T, x) : pyconvert_unconverted()
