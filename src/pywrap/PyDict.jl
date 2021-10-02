"""
    PyDict{K=Py,V=Py}([x])

Wraps the Python dict `x` (or anything satisfying the mapping interface) as an `AbstractDict{K,V}`.

If `x` is not a Python object, it is converted to one using `pydict`.
"""
struct PyDict{K,V} <: AbstractDict{K,V}
    py :: Py
    PyDict{K,V}(::Val{:new}, py::Py) where {K,V} = new{K,V}(py)
end
export PyDict

PyDict{K,V}(x=pydict()) where {K,V} = PyDict{K,V}(Val(:new), ispy(x) ? Py(x) : pydict(x))
PyDict{K}(x=pydict()) where {K} = PyDict{K,Py}(x)
PyDict(x=pydict()) = PyDict{Py,Py}(x)

ispy(::PyDict) = true
getpy(x::PyDict) = x.py
pydel!(x::PyDict) = pydel!(x.py)

pyconvert_rule_mapping(::Type{T}, x::Py, ::Type{PyDict{K,V}}=Utils._type_ub(T)) where {T<:PyDict,K,V} =
    if PyDict{Py,Py} <: T
        pyconvert_return(PyDict{Py,Py}(x))
    elseif PyDict{K,Py} <: T
        pyconvert_return(PyDict{K,Py}(x))
    elseif PyDict{Py,V} <: T
        pyconvert_return(PyDict{Py,V}(x))
    else
        pyconvert_return(PyDict{K,V}(x))
    end

Base.length(x::PyDict) = Int(pylen(x))

function Base.iterate(x::PyDict{K,V}, it::Py=pyiter(x)) where {K,V}
    k_ = unsafe_pynext(it)
    ispynull(k_) && return nothing
    v_ = pygetitem(x, k_)
    k = pyconvert_and_del(K, k_)
    v = pyconvert_and_del(V, v_)
    return (k => v, it)
end

function Base.iterate(x::Base.KeySet{K,PyDict{K,V}}, it::Py=pyiter(x.dict)) where {K,V}
    k_ = unsafe_pynext(it)
    ispynull(k_) && return nothing
    k = pyconvert_and_del(K, k_)
    return (k, it)
end

function Base.getindex(x::PyDict{K,V}, k) where {K,V}
    return pyconvert_and_del(V, pygetitem(x, convert(K, k)))
end

function Base.setindex!(x::PyDict{K,V}, v, k) where {K,V}
    pysetitem(x, convert(K, k), convert(V, v))
    return x
end

function Base.delete!(x::PyDict{K,V}, k) where {K,V}
    r = pyconvert_tryconvert(K, k)
    if !pyconvert_isunconverted(r)
        pydelitem(x, pyconvert_result(K, r))
    end
    return x
end

function Base.empty!(x::PyDict)
    pydel!(@py x.clear())
    return x
end

function Base.copy(x::PyDict{K,V}) where {K,V}
    PyDict{K,V}(Val(:new), @py x.copy())
end

function Base.haskey(x::PyDict{K,V}, k) where {K,V}
    r = pyconvert_tryconvert(K, k)
    if pyconvert_isunconverted(r)
        return false
    else
        return pycontains(x, pyconvert_result(K, r))
    end
end

function Base.get(x::PyDict, k, d)
    if haskey(x, k)
        return x[k]
    else
        return d
    end
end

function Base.get(f::Union{Function,Type}, x::PyDict, k)
    if haskey(x, k)
        return x[k]
    else
        return f()
    end
end

function Base.get!(x::PyDict{K,V}, k, d) where {K,V}
    if haskey(x, k)
        return x[k]
    else
        return x[k] = convert(V, d)
    end
end

function Base.get!(f::Union{Function,Type}, x::PyDict{K,V}, k) where {K,V}
    if haskey(x, k)
        return x[k]
    else
        return x[k] = convert(V, f())
    end
end
