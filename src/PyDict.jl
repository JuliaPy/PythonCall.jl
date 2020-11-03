struct PyDict{K,V} <: AbstractDict{K,V}
    o :: PyObject
    PyDict{K,V}(o::AbstractPyObject) where {K,V} = new{K,V}(PyObject(o))
end
PyDict{K,V}(o=pydict()) where {K,V} = PyDict{K,V}(pydict(o))
PyDict{K}(o=pydict()) where {K} = PyDict{K,PyObject}(o)
PyDict(o=pydict()) = PyDict{PyObject}(o)
export PyDict

pyobject(x::PyDict) = x.o

function Base.iterate(x::PyDict{K,V}, _it=nothing) where {K,V}
    it = _it===nothing ? pyiter(x.o.items()) : _it
    ptr = cpycall_raw(Val(:PyIter_Next), CPyPtr, it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        kv = pynewobject(ptr)
        (convert(K, kv[0]), convert(V, kv[1])), it
    end
end

Base.setindex!(x::PyDict{K,V}, v, k) where {K,V} =
    (pysetitem(x.o, convert(K, k), convert(V, v)); x)

Base.getindex(x::PyDict{K,V}, k) where {K,V} =
    pygetitem(x.o, convert(K, k))

Base.delete!(x::PyDict{K,V}, k) where {K,V} =
    (pydelitem(x.o, convert(K, k)); x)

Base.length(x::PyDict) = Int(pylen(x.o))
