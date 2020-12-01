"""
    PyDict{K=PyObject, V=PyObject}(o=pydict())

Wrap the Python dictionary `o` (or anything satisfying the mapping interface) as a Julia dictionary with keys of type `K` and values of type `V`.
"""
struct PyDict{K,V} <: AbstractDict{K,V}
    o :: PyObject
    PyDict{K,V}(o::AbstractPyObject) where {K,V} = new{K,V}(PyObject(o))
end
PyDict{K,V}(o=pydict()) where {K,V} = PyDict{K,V}(pydict(o))
PyDict{K}(o=pydict()) where {K} = PyDict{K,PyObject}(o)
PyDict(o=pydict()) = PyDict{PyObject}(o)
export PyDict

pyobject(x::PyDict) = x.o

function Base.iterate(x::PyDict{K,V}, it=pyiter(x.o.items())) where {K,V}
    ptr = C.PyIter_Next(it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        kv = pynewobject(ptr)
        (pyconvert(K, kv[0]) => pyconvert(V, kv[1])), it
    end
end

Base.setindex!(x::PyDict{K,V}, v, k) where {K,V} =
    (pysetitem(x.o, convert(K, k), convert(V, v)); x)

Base.getindex(x::PyDict{K,V}, k) where {K,V} =
    pyconvert(V, pygetitem(x.o, convert(K, k)))

Base.delete!(x::PyDict{K,V}, k) where {K,V} =
    (pydelitem(x.o, convert(K, k)); x)

Base.length(x::PyDict) = Int(pylen(x.o))

Base.empty!(x::PyDict) = (x.o.clear(); x)

Base.copy(x::PyDict) = typeof(x)(x.o.copy())

function Base.get(x::PyDict{K,V}, _k, d) where {K,V}
    k = convert(K, _k)
    pycontains(x.o, k) ? x[k] : d
end

function Base.get(f::Function, x::PyDict{K,V}, _k) where {K,V}
    k = convert(K, _k)
    pycontains(x.o, k) ? x[k] : f()
end

function Base.get!(x::PyDict{K,V}, _k, d) where {K,V}
    k = convert(K, _k)
    pycontains(x.o, k) ? x[k] : (x[k] = d)
end

function Base.get!(f::Function, x::PyDict{K,V}, _k) where {K,V}
    k = convert(K, _k)
    pycontains(x.o, k) ? x[k] : (x[k] = f())
end
