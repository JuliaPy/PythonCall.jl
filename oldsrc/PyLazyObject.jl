export PyLazyObject

ispyreftype(::Type{<:PyLazyObject}) = true
pyptr(x::PyLazyObject) = begin
    ptr = CPyPtr(x.ptr)
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
