"""
    PyIterable{T=Py}(x)

This object iterates over iterable Python object `x`, yielding values of type `T`.
"""
struct PyIterable{T}
    py :: Py
    PyIterable{T}(x) where {T} = new{T}(Py(x))
end
export PyIterable

PyIterable(x) = PyIterable{Py}(x)

ispy(x::PyIterable) = true
Py(x::PyIterable) = x.py

Base.IteratorSize(::Type{PyIterable{T}}) where {T} = Base.SizeUnknown()
Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it::Py=pyiter(x)) where {T}
    y = unsafe_pynext(it)
    if pyisnew(y)
        unsafe_pydel!(it)
        return nothing
    else
        return (pyconvert(T, y), it)
    end
end

function pyconvert_rule_iterable(::Type{T}, x::Py, ::Type{T1}=Utils._type_ub(T)) where {T<:PyIterable,T1}
    pyconvert_return(T1(x))
end
