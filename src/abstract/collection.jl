function _pyconvert_rule_iterable(ans::Vector{T0}, it::Py, ::Type{T1}) where {T0,T1}
    @label again
    x_ = pynext(it)
    if ispynull(x_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    r = pytryconvert(T1, x_)
    pydel!(x_)
    if pyconvert_isunconverted(r)
        return r
    end
    x = pyconvert_result(r)
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
