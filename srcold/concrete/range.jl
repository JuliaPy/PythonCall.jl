
function pyconvert_rule_range(::Type{R}, x::Py, ::Type{StepRange{T0,S0}}=Utils._type_lb(R), ::Type{StepRange{T1,S1}}=Utils._type_ub(R)) where {R<:StepRange,T0,S0,T1,S1}
    a = @pyconvert(Utils._typeintersect(Integer, T1), x.start)
    b = @pyconvert(Utils._typeintersect(Integer, S1), x.step)
    c = @pyconvert(Utils._typeintersect(Integer, T1), x.stop)
    a′, c′ = promote(a, c - oftype(c, sign(b)))
    T2 = Utils._promote_type_bounded(T0, typeof(a′), typeof(c′), T1)
    S2 = Utils._promote_type_bounded(S0, typeof(c′), S1)
    pyconvert_return(StepRange{T2, S2}(a′, b, c′))
end

function pyconvert_rule_range(::Type{R}, x::Py, ::Type{UnitRange{T0}}=Utils._type_lb(R), ::Type{UnitRange{T1}}=Utils._type_ub(R)) where {R<:UnitRange,T0,T1}
    b = @pyconvert(Int, x.step)
    b == 1 || return pyconvert_unconverted()
    a = @pyconvert(Utils._typeintersect(Integer, T1), x.start)
    c = @pyconvert(Utils._typeintersect(Integer, T1), x.stop)
    a′, c′ = promote(a, c - oftype(c, 1))
    T2 = Utils._promote_type_bounded(T0, typeof(a′), typeof(c′), T1)
    pyconvert_return(UnitRange{T2}(a′, c′))
end
