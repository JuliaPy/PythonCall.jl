"""
    PyIterable{T=PyObject}(o)

Wrap the Python object `o` into a Julia object which iterates values of type `T`.
"""
struct PyIterable{T}
    o :: PyObject
    PyIterable{T}(o) where {T} = new{T}(PyObject(o))
end
PyIterable(o) = PyIterable{PyObject}(o)
export PyIterable

pyobject(x::PyIterable) = x.o

Base.length(x::PyIterable) = Int(pylen(x.o))

Base.IteratorSize(::Type{<:PyIterable}) = Base.SizeUnknown()

Base.IteratorEltype(::Type{<:PyIterable}) = Base.HasEltype()

Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it=pyiter(x.o)) where {T}
    ptr = cpycall_raw(Val(:PyIter_Next), CPyPtr, it)
    if ptr == C_NULL
        pyerrcheck()
        nothing
    else
        (pyconvert(T, pynewobject(ptr)), it)
    end
end
