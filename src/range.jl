const pyrangetype = pylazyobject(() -> pybuiltins.range)
export pyrangetype

pyrange(args...; opts...) = pyrangetype(args...; opts...)
pyrange(x::AbstractRange{<:Integer}) = pyrange(first(x), last(x)+sign(step(x)), step(x))
export pyrange

pyisrange(o::PyObject) = pytypecheck(o, pyrangetype)
export pyisrange

function pyrange_tryconvert(::Type{T}, o::PyObject) where {A<:Integer, C<:Integer, T<:StepRange{A,C}}
    a = pyint_tryconvert(A, o.start)
    a === PyConvertFail() && return a
    b = pyint_tryconvert(A, o.stop)
    b === PyConvertFail() && return b
    c = pyint_tryconvert(C, o.step)
    c === PyConvertFail() && return c
    return StepRange{A,C}(a, c, b-oftype(b, sign(c)))
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {A<:Integer, T<:UnitRange{A}}
    o.step == pyint(1) || return PyConvertFail()
    a = pyint_tryconvert(A, o.start)
    a === PyConvertFail() && return a
    b = pyint_tryconvert(A, o.stop)
    b === PyConvertFail() && return b
    return UnitRange{A}(a, b-one(b))
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {T<:StepRange}
    tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {T<:UnitRange}
    tryconvert(T, pyrange_tryconvert(UnitRange{BigInt}, o))
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {I<:Integer, T<:AbstractRange{I}}
    if (S = _typeintersect(T, StepRange{I,I})) != Union{}
        pyrange_tryconvert(S, o)
    elseif (S = _typeintersect(T, UnitRange{I})) != Union{}
        pyrange_tryconvert(S, o)
    else
        tryconvert(T, pyrange_tryconvert(StepRange{I, I}, o))
    end
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {T<:AbstractRange{<:Integer}}
    if (S = _typeintersect(T, StepRange{<:Integer, <:Integer})) != Union{}
        pyrange_tryconvert(S, o)
    elseif (S = _typeintersect(T, UnitRange{<:Integer})) != Union{}
        pyrange_tryconvert(S, o)
    else
        tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
    end
end

function pyrange_tryconvert(::Type{T}, o::PyObject) where {T}
    if (S = _typeintersect(T, AbstractRange{<:Integer})) != Union{}
        pyrange_tryconvert(S, o)
    else
        tryconvert(T, pyrange_tryconvert(StepRange{BigInt, BigInt}, o))
    end
end
