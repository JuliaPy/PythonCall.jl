function explode_union(T)
    @nospecialize T

    # unpack unionall
    S = T
    vars = []
    while S isa UnionAll
        pushfirst!(vars, S.var)
        S = S.body
    end

    if S isa Union
        Us = Any[explode_union(S.a)..., explode_union(S.b)...]
        Any[foldl((body, var) -> UnionAll(var, body), vars, init = U) for U in Us]
    elseif S == Union{}
        Any[]
    else
        Any[T]
    end
end

@generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)

@generated _type_ub(::Type{T}) where {T} = begin
    S = T
    while S isa UnionAll
        S = S{S.var.ub}
    end
    S
end

@generated _type_lb(::Type{T}) where {T} = begin
    R = _unwrap_unionall(T)
    if R isa DataType
        S = T
        while S isa UnionAll
            S = S{S.var in R.parameters ? S.var.lb : S.var.ub}
        end
        S
    else
        _type_ub(T)
    end
end

@generated function _unwrap_unionall(::Type{T}) where {T}
    R = T
    while R isa UnionAll
        R = R.body
    end
    R
end

@generated _promote_type_bounded(::Type{S}, ::Type{T}, ::Type{B}) where {S,T,B} = begin
    S <: B || error("require S <: B")
    T <: B || error("require T <: B")
    if B isa Union
        return Union{
            _promote_type_bounded(typeintersect(S, B.a), typeintersect(T, B.a), B.a),
            _promote_type_bounded(typeintersect(S, B.b), typeintersect(T, B.b), B.b),
        }
    else
        R = promote_type(S, T)
        if R <: B
            return R
        else
            R = typeintersect(typejoin(S, T), B)
            if R <: B
                return R
            else
                return B
            end
        end
    end
end

@generated _promote_type_bounded(
    ::Type{T1},
    ::Type{T2},
    ::Type{T3},
    ::Type{B},
) where {T1,T2,T3,B} = _promote_type_bounded(_promote_type_bounded(T1, T2, B), T3, B)
