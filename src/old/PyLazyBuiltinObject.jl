export PyLazyBuiltinObject

ispyreftype(::Type{<:PyLazyBuiltinObject}) = true
pyptr(x::PyLazyBuiltinObject) = begin
    ptr = CPyPtr(x.ptr)
    if isnull(ptr)
        x.ptr = ptr = C.PyDict_GetItemString(C.PyEval_GetBuiltins(), x.name)
        C.Py_IncRef(ptr)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyLazyBuiltinObject) = checknull(pyptr(x))
Base.show(io::IO, x::PyLazyBuiltinObject) = begin
    show(io, typeof(x))
    print(io, "(")
    show(io, x.name)
    print(io, ")")
end
