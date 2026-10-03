PyTuple(x = pytuple()) = PyTuple{Tuple}(x)

ispy(::PyTuple) = true
Py(x::PyTuple) = x.py

function pyconvert_rule_pytuple(
    ::Type{T},
    x::Py,
    ::Type{T1} = Utils._type_ub(T),
) where {T<:PyTuple,T1}
    pyconvert_return(T1(x))
end

Base.IteratorSize(::Type{<:PyTuple}) = Base.HasLength()
Base.IteratorEltype(::Type{<:PyTuple}) = Base.HasEltype()
Base.eltype(::Type{PyTuple{T}}) where {T} = eltype(T)

Base.length(x::PyTuple{T}) where {T} = Base.isvatuple(T) ? Int(pylen(x)) : fieldcount(T)
Base.size(x::PyTuple) = (length(x),)

Base.@propagate_inbounds function Base.getindex(x::PyTuple{T}, i::Int) where {T}
    @boundscheck (1 <= i <= length(x) || throw(BoundsError(x, i)))
    return pyconvert(fieldtype(T, i), @py x[@jl(i - 1)])
end

function Base.iterate(x::PyTuple, i::Int = 1)
    i > length(x) && return nothing
    return (@inbounds(x[i]), i + 1)
end

Base.Tuple(x::PyTuple{T}) where {T} = pyconvert(T, x.py)
