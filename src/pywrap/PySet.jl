"""
    PySet{T=Py}([x])

Wraps the Python set `x` (or anything satisfying the set interface) as an `AbstractSet{T}`.

If `x` is not a Python object, it is converted to one using `pyset`.
"""
struct PySet{T} <: AbstractSet{T}
    py :: Py
    PySet{T}(::Val{:new}, py::Py) where {T} = new{T}(py)
end
export PySet

PySet{T}(x=pyset()) where {T} = PySet{T}(Val(:new), ispy(x) ? Py(x) : pyset(x))
PySet(x=pyset()) = PySet{Py}(x)

ispy(::PySet) = true
getpy(x::PySet) = x.py
pydel!(x::PySet) = pydel!(x.py)

pyconvert_rule_set(::Type{T}, x::Py) where {T<:PySet} = pyconvert_return(Utils._type_ub(T)(x))

Base.length(x::PySet) = Int(pylen(x.py))

Base.isempty(x::PySet) = length(x) == 0

function Base.iterate(x::PySet{T}, it::Py=pyiter(x)) where {T}
    y = pynext(it)
    if ispynull(y)
        pydel!(it)
        return nothing
    else
        return (pyconvert_and_del(T, y), it)
    end
end

function Base.in(v::T, x::PySet{T}) where {T}
    # TODO: how to handle general v?
    pycontains(x, v)
end

function Base.push!(x::PySet{T}, v) where {T}
    pydel!(@py x.add(@jl(convert(T, v))))
    return x
end

function Base.delete!(x::PySet{T}, v::T) where {T}
    # TODO: how to handle general v?
    pydel!(@py x.discard(v))
    return x
end

Base.@propagate_inbounds function Base.pop!(x::PySet{T}) where {T}
    @boundscheck (isempty(x) && throw(ArgumentError("set must be non-empty")))
    return pyconvert_and_del(T, @py x.pop())
end

function Base.pop!(x::PySet{T}, v::T) where {T}
    # TODO: how to handle general v?
    if v in x
        delete!(x, v)
        return v
    else
        throw(KeyError(v))
    end
end

function Base.pop!(x::PySet{T}, v::T, d) where {T}
    # TODO: how to handle general v?
    if v in x
        delete!(x, v)
        return v
    else
        return d
    end
end

function Base.empty!(x::PySet)
    pydel!(@py x.clear())
    return x
end

function Base.copy(x::PySet{T}) where {T}
    return PySet{T}(Val(:new), @py x.copy())
end
