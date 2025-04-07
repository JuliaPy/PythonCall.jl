PyIterable(x) = PyIterable{Py}(x)

ispy(x::PyIterable) = true
Py(x::PyIterable) = x.py

Base.IteratorSize(::Type{PyIterable{T}}) where {T} = Base.SizeUnknown()
Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it::Py = pyiter(x)) where {T}
    y = unsafe_pynext(it)
    if pyisnull(y)
        pydel!(it)
        return nothing
    else
        return (pyconvert(T, y), it)
    end
end

function pyconvert_rule_iterable(
    ::Type{T},
    x::Py,
    ::Type{T1} = Utils.type_ub(T),
) where {T<:PyIterable,T1}
    pyconvert_return(T1(x))
end
