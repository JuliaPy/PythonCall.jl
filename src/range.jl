const pyrangetype = PyLazyObject(() -> pybuiltins.range)
export pyrangetype

pyrange(args...; opts...) = pyrangetype(args...; opts...)
pyrange(x::AbstractRange{<:Integer}) = pyrange(first(x), last(x)+sign(step(x)), step(x))
export pyrange

pyisrange(o::AbstractPyObject) = pytypecheck(o, pyrangetype)
export pyisrange

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {A<:Integer, C<:Integer, T<:StepRange{A,C}}
    a = pyint_tryconvert(A, o.start)
    a === PyConvertFail() && return a
    b = pyint_tryconvert(A, o.stop)
    b === PyConvertFail() && return b
    c = pyint_tryconvert(C, o.step)
    c === PyConvertFail() && return c
    return StepRange{A,C}(a, c, b-oftype(b, sign(c)))
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {A<:Integer, T<:UnitRange{A}}
    o.step == pyint(1) || return PyConvertFail()
    a = pyint_tryconvert(A, o.start)
    a === PyConvertFail() && return a
    b = pyint_tryconvert(A, o.stop)
    b === PyConvertFail() && return b
    return UnitRange{A}(a, b-one(b))
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:StepRange}
    tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:UnitRange}
    tryconvert(T, pyrange_tryconvert(UnitRange{BigInt}, o))
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {I<:Integer, T<:AbstractRange{I}}
    if (S = _typeintersect(T, StepRange{I,I})) != Union{}
        @info "C1"
        pyrange_tryconvert(S, o)
    elseif (S = _typeintersect(T, UnitRange{I})) != Union{}
        @info "C3"
        pyrange_tryconvert(S, o)
    else
        @info "C2"
        tryconvert(T, pyrange_tryconvert(StepRange{I, I}, o))
    end
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:AbstractRange{<:Integer}}
    if (S = _typeintersect(T, StepRange{<:Integer, <:Integer})) != Union{}
        @info "B1"
        pyrange_tryconvert(S, o)
    elseif (S = _typeintersect(T, UnitRange{<:Integer})) != Union{}
        @info "B3"
        pyrange_tryconvert(S, o)
    else
        @info "B2"
        tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
    end
end

function pyrange_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if (S = _typeintersect(T, AbstractRange{<:Integer})) != Union{}
        @info "A1"
        pyrange_tryconvert(S, o)
    else
        @info "A2"
        tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
    end
end
