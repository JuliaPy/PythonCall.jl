"""
    PyIterable{T=PyObject}(o)

Wrap the Python object `o` into a Julia object which iterates values of type `T`.
"""
struct PyIterable{T}
    ref :: PyRef
    PyIterable{T}(o) where {T} = new{T}(PyRef(o))
end
PyIterable(o) = PyIterable{PyObject}(o)
export PyIterable

ispyreftype(::Type{<:PyIterable}) = true
pyptr(x::PyIterable) = pyptr(x.ref)
Base.unsafe_convert(::Type{CPyPtr}, x::PyIterable) = checknull(pyptr(x))
C.PyObject_TryConvert__initial(o, ::Type{T}) where {T<:PyIterable} = C.putresult(T(pyborrowedref(o)))

Base.length(x::PyIterable) = Int(pylen(x))

Base.IteratorSize(::Type{<:PyIterable}) = Base.SizeUnknown()

Base.IteratorEltype(::Type{<:PyIterable}) = Base.HasEltype()

Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it=pyiter(PyRef, x)) where {T}
    vo = C.PyIter_Next(it)
    if !isnull(vo)
        r = C.PyObject_Convert(vo, T)
        C.Py_DecRef(vo)
        checkm1(r)
        (takeresult(T), it)
    elseif C.PyErr_IsSet()
        pythrow()
    else
        nothing
    end
end
