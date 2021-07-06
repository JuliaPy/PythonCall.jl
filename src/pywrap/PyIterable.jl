"""
    PyIterable{T=Py}(x)

This object iterates over iterable Python object `x`, yielding values of type `T`.
"""
struct PyIterable{T}
    py :: Py
    PyIterable{T}(::Val{:new}, py::Py) where {T} = new{T}(py)
end
export PyIterable

PyIterable{T}(x) where {T} = PyIterable{T}(Val(:new), Py(x))
PyIterable(x) = PyIterable{Py}(x)

ispy(x::PyIterable) = true
getpy(x::PyIterable) = x.py
pydel!(x::PyIterable) = pydel!(x.py)

Base.IteratorSize(::Type{PyIterable{T}}) where {T} = Base.SizeUnknown()
Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it::Py=pyiter(x.py)) where {T}
    y = pynext(it)
    if ispynull(y)
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
