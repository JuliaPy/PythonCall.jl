"""
    PyLazyObject(x)

Convert `x` to a Python object lazily, i.e. the conversion happens the first time the object is accessed.
"""
mutable struct PyLazyObject{T}
    ptr::CPyPtr
    val::T
    PyLazyObject(val::T) where {T} = finalizer(pyref_finalize!, new{T}(CPyPtr(0), val))
end
export PyLazyObject

ispyreftype(::Type{<:PyLazyObject}) = true
pyptr(x::PyLazyObject) = begin
    ptr = x.ptr
    if isnull(ptr)
        x.ptr = ptr = C.PyObject_From(x.val)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyLazyObject) = checknull(pyptr(x))
Base.show(io::IO, x::PyLazyObject) = begin
    show(io, typeof(x))
    print(io, "(")
    show(io, x.val)
    print(io, ")")
end
