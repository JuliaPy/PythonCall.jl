export PyInternedString

ispyreftype(::Type{PyInternedString}) = true
pyptr(x::PyInternedString) = begin
    ptr = CPyPtr(x.ptr)
    if isnull(ptr)
        s = Ref{CPyPtr}()
        s[] = C.PyUnicode_From(x.val)
        isnull(s[]) && return ptr
        C.PyUnicode_InternInPlace(s)
        ptr = x.ptr = s[]
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyInternedString) = checknull(pyptr(x))
Base.show(io::IO, x::PyInternedString) = begin
    show(io, typeof(x))
    print(io, '(')
    show(io, x.val)
    print(io, ')')
end

"""
    pystr"..." :: PyInternedString

Literal syntax for an interned Python string.
"""
macro pystr_str(s::String)
    PyInternedString(s)
end
export @pystr_str
