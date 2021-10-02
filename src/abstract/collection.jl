# Vector

function _pyconvert_rule_iterable(ans::Vector{T0}, it::Py, ::Type{T1}) where {T0,T1}
    @label again
    x_ = unsafe_pynext(it)
    if ispynull(x_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    x = @pyconvert_and_del(T1, x_)
    if x isa T0
        push!(ans, x)
        @goto again
    end
    T2 = Utils._promote_type_bounded(T0, typeof(x), T1)
    ans2 = Vector{T2}(ans)
    push!(ans2, x)
    return _pyconvert_rule_iterable(ans2, it, T1)
end

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Vector{T0}}=Utils._type_lb(R), ::Type{Vector{T1}}=Utils._type_ub(R)) where {R<:Vector,T0,T1}
    it = pyiter(x)
    ans = Vector{T0}()
    return _pyconvert_rule_iterable(ans, it, T1)
end

# Set

function _pyconvert_rule_iterable(ans::Set{T0}, it::Py, ::Type{T1}) where {T0,T1}
    @label again
    x_ = unsafe_pynext(it)
    if ispynull(x_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    x = @pyconvert_and_del(T1, x_)
    if x isa T0
        push!(ans, x)
        @goto again
    end
    T2 = Utils._promote_type_bounded(T0, typeof(x), T1)
    ans2 = Set{T2}(ans)
    push!(ans2, x)
    return _pyconvert_rule_iterable(ans2, it, T1)
end

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Set{T0}}=Utils._type_lb(R), ::Type{Set{T1}}=Utils._type_ub(R)) where {R<:Set,T0,T1}
    it = pyiter(x)
    ans = Set{T0}()
    return _pyconvert_rule_iterable(ans, it, T1)
end

# Dict

function _pyconvert_rule_mapping(ans::Dict{K0,V0}, x::Py, it::Py, ::Type{K1}, ::Type{V1}) where {K0,V0,K1,V1}
    @label again
    k_ = unsafe_pynext(it)
    if ispynull(k_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    v_ = pygetitem(x, k_)
    k = @pyconvert_and_del(K1, k_)
    v = @pyconvert_and_del(V1, v_)
    if k isa K0 && v isa V0
        push!(ans, k => v)
        @goto again
    end
    K2 = Utils._promote_type_bounded(K0, typeof(k), K1)
    V2 = Utils._promote_type_bounded(V0, typeof(v), V1)
    ans2 = Dict{K2,V2}(ans)
    push!(ans2, k => v)
    return _pyconvert_rule_mapping(ans2, x, it, K1, V1)
end

function pyconvert_rule_mapping(::Type{R}, x::Py, ::Type{Dict{K0,V0}}=Utils._type_lb(R), ::Type{Dict{K1,V1}}=Utils._type_ub(R)) where {R<:Dict,K0,V0,K1,V1}
    it = pyiter(x)
    ans = Dict{K0,V0}()
    return _pyconvert_rule_mapping(ans, x, it, K1, V1)
end

# Tuple

function pyconvert_rule_iterable(::Type{T}, xs::Py) where {T<:Tuple}
    T isa DataType || return pyconvert_unconverted()
    ts = collect(T.parameters)
    if !isempty(ts) && Base.isvarargtype(ts[end])
        isvararg = true
        vartype = ts[end].body.parameters[1]::Type
        pop!(ts)
    else
        isvararg = false
        vartype = Union{}
    end
    zs = Any[]
    for x in xs
        if length(zs) < length(ts)
            t = ts[length(zs) + 1]
        elseif isvararg
            t = vartype
        else
            return pyconvert_unconverted()
        end
        z = @pyconvert_and_del(t, x)
        push!(zs, z)
    end
    return length(zs) < length(ts) ? pyconvert_unconverted() : pyconvert_return(T(zs))
end

# N-Tuple for N up to 16
# TODO: Vararg

for N in 0:16
    Ts = [Symbol("T", n) for n in 1:N]
    zs = [Symbol("z", n) for n in 1:N]
    @eval function pyconvert_rule_iterable(::Type{Tuple{$(Ts...)}}, xs::Py) where {$(Ts...)}
        pylen(xs) == $N || return pyconvert_unconverted()
        $((
            :($z = @pyconvert_and_del($T, pytuple_getitem(xs, $(i-1))))
            for (i, T, z) in zip(1:N, Ts, zs)
        )...)
        return pyconvert_return(($(zs...),))
    end
end

# Pair

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Pair{K0,V0}}=Utils._type_lb(R), ::Type{Pair{K1,V1}}=Utils._type_ub(R)) where {R<:Pair,K0,V0,K1,V1}
    it = pyiter(x)
    k_ = unsafe_pynext(it)
    if ispynull(k_)
        pydel!(it)
        pydel!(k_)
        return pyconvert_unconverted()
    end
    k = @pyconvert_and_del(K1, k_)
    v_ = unsafe_pynext(it)
    if ispynull(v_)
        pydel!(it)
        pydel!(v_)
        return pyconvert_unconverted()
    end
    v = @pyconvert_and_del(V1, v_)
    z_ = unsafe_pynext(it)
    pydel!(it)
    if ispynull(z_)
        pydel!(z_)
    else
        pydel!(z_)
        return pyconvert_unconverted()
    end
    K2 = Utils._promote_type_bounded(K0, typeof(k), K1)
    V2 = Utils._promote_type_bounded(V0, typeof(v), V1)
    return pyconvert_return(Pair{K2,V2}(k, v))
end
