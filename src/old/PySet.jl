"""
    PySet{T=PyObject}(o=pyset())

Wrap the Python set `o` (or anything satisfying the set interface) as a Julia set with elements of type `T`.
"""
struct PySet{T} <: AbstractSet{T}
    o :: PyObject
    PySet{T}(o::PyObject) where {T} = new{T}(o)
end
PySet{T}(o=pyset()) where {T} = PySet{T}(pyset(o))
PySet(o=pyset()) = PySet{PyObject}(o)
export PySet

pyobject(x::PySet) = x.o

function Base.iterate(x::PySet{T}, it=pyiter(x.o)) where {T}
    ptr = C.PyIter_Next(it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        (pyconvert(T, pynewobject(ptr)), it)
    end
end

Base.length(x::PySet) = Int(pylen(x.o))

function Base.in(_v, x::PySet{T}) where {T}
    v = tryconvert(T, _v)
    v === PyConvertFail() ? false : pycontains(x, v)
end

function Base.push!(x::PySet{T}, v) where {T}
    x.o.add(convert(T, v))
    x
end

function Base.delete!(x::PySet{T}, _v) where {T}
    v = tryconvert(T, _v)
    v === PyConvertFail() || x.o.discard(v)
    x
end

function Base.pop!(x::PySet{T}) where {T}
    v = x.o.pop()
    pyconvert(T, v)
end

function Base.pop!(x::PySet{T}, _v) where {T}
    v = convert(T, _v)
    x.o.remove(v)
    v
end

function Base.pop!(x::PySet{T}, _v, d) where {T}
    v = tryconvert(T, _v)
    (v !== PyConvertFail() && v in x) ? pop!(x, v) : d
end

function Base.filter!(f, x::PySet{T}) where {T}
    d = pylist()
    for _v in x.o
        if !f(pyconvert(T, _v))
            d.append(_v)
        end
    end
    x.o.difference_update(d)
    x
end

Base.empty!(x::PySet) = (x.o.clear(); x)

Base.copy(x::PySet) = typeof(x)(x.o.copy())
