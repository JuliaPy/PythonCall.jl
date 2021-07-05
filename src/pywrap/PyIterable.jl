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
(::Type{T})(x) where {T<:PyIterable} = Utils._type_ub(T)(x)

ispy(x::PyIterable) = true
getpy(x::PyIterable) = x.py
pydel!(x::PyIterable) = pydel!(x.py)

Base.IteratorSize(::Type{PyIterable{T}}) where {T} = Base.SizeUnknown()
Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it::Py=pyiter(x.py)) where {T}
    y_ = pynext(it)
    if ispynull(y_)
        pydel!(it)
        return nothing
    else
        y = pyconvert(T, y_)
        pydel!(y_)
        return (y, it)
    end
end

pyconvert_rule_iterable(::Type{T}, x::Py) where {T<:PyIterable} = pyconvert_return(T(x))
