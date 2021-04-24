export PyCode

ispyreftype(::Type{PyCode}) = true
pyptr(co::PyCode) = begin
    ptr = CPyPtr(co.ptr)
    if isnull(ptr)
        ptr = co.ptr = C.Py_CompileString(
            co.code,
            co.filename,
            co.mode == :exec ? C.Py_file_input :
            co.mode == :eval ? C.Py_eval_input : error("invalid mode $(repr(co.mode))"),
        )
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyCode) = checknull(pyptr(x))
Base.show(io::IO, x::PyCode) = begin
    show(io, typeof(x))
    print(io, "(")
    show(io, x.code)
    print(io, ", ")
    show(io, x.filename)
    print(io, ", ")
    show(io, x.mode)
    print(io, ")")
end
