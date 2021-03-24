"""
    PyCode(code::String, filename::String, mode::Symbol)

A Python code object, representing the compiled contents of `code`.

The `filename` is used for exception printing. The mode must be `:exec` or `:eval`.

See also [`@py_cmd`](@ref) and [`@pyv_cmd`](@ref).
"""
mutable struct PyCode
    ref::PyRef
    code::String
    filename::String
    mode::Symbol
    PyCode(code::String, filename::String, mode::Symbol) = begin
        mode in (:exec, :eval) || error("invalid mode $(repr(mode))")
        new(PyRef(), code, filename, mode)
    end
end
export PyCode

ispyreftype(::Type{PyCode}) = true
pyptr(co::PyCode) = begin
    ptr = co.ref.ptr
    if isnull(ptr)
        ptr =
            co.ref.ptr = C.Py_CompileString(
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
