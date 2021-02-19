"""
    PyInternedString(x::String)

Convert `x` to an interned Python string.

This can provide a performance boost when using strings to index dictionaries or get attributes.

See also [`@pystr_str`].
"""
mutable struct PyInternedString
    ref::PyRef
    val::String
    PyInternedString(x::String) = new(PyRef(), x)
end
export PyInternedString

ispyreftype(::Type{PyInternedString}) = true
pyptr(x::PyInternedString) = begin
    ptr = x.ref.ptr
    if isnull(ptr)
        s = Ref{CPyPtr}()
        s[] = C.PyUnicode_From(x.val)
        isnull(s[]) && return ptr
        C.PyUnicode_InternInPlace(s)
        ptr = x.ref.ptr = s[]
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyInternedString) = checknull(pyptr(x))

"""
    pystr"..." :: PyInternedString

Literal syntax for an interned Python string.
"""
macro pystr_str(s::String)
    PyInternedString(s)
end
export @pystr_str
