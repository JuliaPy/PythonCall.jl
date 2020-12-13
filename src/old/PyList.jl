"""
    PyList{T=PyObject}(o=pylist())

Wrap the Python list `o` (or anything satisfying the sequence interface) as a Julia vector with elements of type `T`.
"""
struct PyList{T} <: AbstractVector{T}
    o :: PyObject
    PyList{T}(o::PyObject) where {T} = new{T}(o)
end
PyList{T}(o=pylist()) where {T} = PyList{T}(pylist(o))
PyList(o=pylist()) = PyList{PyObject}(o)
export PyList

pyobject(x::PyList) = x.o

Base.length(x::PyList) = Int(pylen(x.o))

Base.size(x::PyList) = (length(x),)

Base.getindex(x::PyList{T}, i::Integer) where {T} = pyconvert(T, pygetitem(x.o, i-1))

Base.setindex!(x::PyList{T}, v, i::Integer) where {T} = (pysetitem(x.o, i-1, convert(T, v)); x)

Base.insert!(x::PyList{T}, i::Integer, v) where {T} = (x.o.insert(i-1, convert(T, v)); x)

Base.push!(x::PyList{T}, v) where {T} = (x.o.append(convert(T, v)); x)

Base.pushfirst!(x::PyList, v) = insert!(x, 1, v)

Base.pop!(x::PyList{T}) where {T} = pyconvert(T, x.o.pop())

Base.popat!(x::PyList{T}, i::Integer) where {T} = pyconvert(T, x.o.pop(i-1))

Base.popfirst!(x::PyList) = pop!(x, 1)

Base.reverse!(x::PyList) = (x.o.reverse(); x)

# TODO: support kwarg `by` (becomes python kwarg `key`)
Base.sort!(x::PyList; rev=false) = (x.o.sort(reverse=rev); x)

Base.empty!(x::PyList) = (x.o.clear(); x)

Base.copy(x::PyList) = typeof(x)(x.o.copy())
