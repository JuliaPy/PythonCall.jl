"""
    PyIterable{T=PyObject}(o)

Wrap the Python object `o` into a Julia object which iterates values of type `T`.
"""
mutable struct PyIterable{T}
    ptr::CPyPtr
    PyIterable{T}(::Val{:new}, ptr::Ptr) where {T} = finalizer(pyref_finalize!, new{T}(CPyPtr(ptr)))
end
PyIterable{T}(o) where {T} = PyIterable{T}(Val(:new), checknull(C.PyObject_From(o)))
PyIterable(o) = PyIterable{PyObject}(o)
export PyIterable

ispyreftype(::Type{<:PyIterable}) = true
pyptr(x::PyIterable) = x.ptr
Base.unsafe_convert(::Type{CPyPtr}, x::PyIterable) = checknull(pyptr(x))
C.PyObject_TryConvert__initial(o, ::Type{PyIterable}) =
    C.PyObject_TryConvert__initial(o, PyIterable{PyObject})
C.PyObject_TryConvert__initial(o, ::Type{PyIterable{T}}) where {T} = begin
    C.Py_IncRef(o)
    C.putresult(PyIterable{T}(Val(:new), o))
end

Base.length(x::PyIterable) = Int(pylen(x))

Base.IteratorSize(::Type{<:PyIterable}) = Base.SizeUnknown()

Base.IteratorEltype(::Type{<:PyIterable}) = Base.HasEltype()

Base.eltype(::Type{PyIterable{T}}) where {T} = T

function Base.iterate(x::PyIterable{T}, it::PyRef = pyiter(PyRef, x)) where {T}
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
