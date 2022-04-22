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
    if pyisnull(y)
        pydel!(it)
        return nothing
    else
        return (pyconvert_and_del(T, y), it)
    end
end

pyconvert_rule_iterable(::Type{T}, x::Py, ::Type{PyIterable{V}}=Utils._type_ub(T)) where {T<:PyIterable,V} =
    if PyIterable{Py} <: T
        pyconvert_return(PyIterable{Py}(x))
    else
        pyconvert_return(PyIterable{V}(x))
    end
