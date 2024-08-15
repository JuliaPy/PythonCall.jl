"""
    PySet{T=Py}([x])

Wraps the Python set `x` (or anything satisfying the set interface) as an `AbstractSet{T}`.

If `x` is not a Python object, it is converted to one using `pyset`.
"""
struct PySet{T} <: AbstractSet{T}
    py::Py
    PySet{T}(x = pyset()) where {T} = new{T}(ispy(x) ? Py(x) : pyset(x))
end
export PySet

PySet(x = pyset()) = PySet{Py}(x)

ispy(::PySet) = true
Py(x::PySet) = x.py

function pyconvert_rule_set(
    ::Type{T},
    x::Py,
    ::Type{T1} = Utils._type_ub(T),
) where {T<:PySet,T1}
    pyconvert_return(T1(x))
end

Base.length(x::PySet) = Int(pylen(x))

Base.isempty(x::PySet) = length(x) == 0

function Base.iterate(x::PySet{T}, it::Py = pyiter(x)) where {T}
    y = unsafe_pynext(it)
    if pyisnull(y)
        pydel!(it)
        return nothing
    else
        return (pyconvert(T, y), it)
    end
end

function Base.in(v, x::PySet{T}) where {T}
    if v isa T
        return pycontains(x, v)
    else
        r = pyconvert_tryconvert(T, v)
        if pyconvert_isunconverted(r)
            return false
        else
            return pycontains(x, pyconvert_result(T, r))
        end
    end
end

function Base.push!(x::PySet{T}, v) where {T}
    pydel!(@py x.add(@jl(convert(T, v))))
    return x
end

function Base.delete!(x::PySet{T}, v) where {T}
    if v isa T
        pydel!(@py x.discard(v))
    else
        r = pyconvert_tryconvert(T, v)
        if !pyconvert_isunconverted(r)
            pydel!(@py x.discard(@jl pyconvert_result(T, r)))
        end
    end
    return x
end

Base.@propagate_inbounds function Base.pop!(x::PySet{T}) where {T}
    @boundscheck (isempty(x) && throw(ArgumentError("set must be non-empty")))
    return pyconvert(T, @py x.pop())
end

function Base.pop!(x::PySet, v)
    if v in x
        delete!(x, v)
        return v
    else
        throw(KeyError(v))
    end
end

function Base.pop!(x::PySet, v, d)
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
    return PySet{T}(@py x.copy())
end
