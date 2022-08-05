"""
    PyList{T=Py}([x])

Wraps the Python list `x` (or anything satisfying the sequence interface) as an `AbstractVector{T}`.

If `x` is not a Python object, it is converted to one using `pylist`.
"""
struct PyList{T} <: AbstractVector{T}
    py :: Py
    PyList{T}(x=pylist()) where {T} = new{T}(ispy(x) ? Py(x) : pylist(x))
end
export PyList

PyList(x=pylist()) = PyList{Py}(x)

ispy(::PyList) = true
Py(x::PyList) = x.py

pyconvert_rule_sequence(::Type{T}, x::Py, ::Type{PyList{V}}=Utils._type_ub(T)) where {T<:PyList,V} =
    if PyList{Py} <: T
        pyconvert_return(PyList{Py}(x))
    else
        pyconvert_return(PyList{V}(x))
    end

Base.length(x::PyList) = Int(pylen(x))

Base.size(x::PyList) = (length(x),)

Base.@propagate_inbounds function Base.getindex(x::PyList{T}, i::Int) where {T}
    @boundscheck checkbounds(x, i)
    return pyconvert(T, @py x[@jl(i-1)])
end

Base.@propagate_inbounds function Base.setindex!(x::PyList{T}, v, i::Int) where {T}
    @boundscheck checkbounds(x, i)
    pysetitem(x, i-1, convert(T, v))
    return x
end

Base.@propagate_inbounds function Base.insert!(x::PyList{T}, i::Integer, v) where {T}
    @boundscheck (i==length(x)+1 || checkbounds(x, i))
    pydel!(@py x.insert(@jl(i-1), @jl(convert(T, v))))
    return x
end

function Base.push!(x::PyList{T}, v) where {T}
    pydel!(@py x.append(@jl(convert(T, v))))
    return x
end

function Base.pushfirst!(x::PyList, v)
    return @inbounds Base.insert!(x, 1, v)
end

function Base.append!(x::PyList, vs)
    for v in vs
        push!(x, v)
    end
    return x
end

function Base.push!(x::PyList, v1, v2, vs...)
    push!(x, v1)
    push!(x, v2, vs...)
end

Base.@propagate_inbounds function Base.pop!(x::PyList{T}) where {T}
    @boundscheck (isempty(x) && throw(BoundsError(x)))
    return pyconvert(T, @py x.pop())
end

Base.@propagate_inbounds function Base.popat!(x::PyList{T}, i::Integer) where {T}
    @boundscheck checkbounds(x, i)
    return pyconvert(T, @py x.pop(@jl(i-1)))
end

Base.@propagate_inbounds function Base.popfirst!(x::PyList{T}) where {T}
    @boundscheck checkbounds(x, 1)
    return pyconvert(T, @py x.pop(0))
end

function Base.reverse!(x::PyList)
    pydel!(@py x.reverse())
    return x
end

function Base.empty!(x::PyList)
    pydel!(@py x.clear())
    return x
end

function Base.copy(x::PyList{T}) where {T}
    return PyList{T}(@py x.copy())
end
