pystr_fromUTF8(x::Ptr, n::Integer) = pynew(errcheck(C.PyUnicode_DecodeUTF8(x, n, C_NULL)))
pystr_fromUTF8(x) = pystr_fromUTF8(pointer(x), sizeof(x))

pystr(x) = pynew(errcheck(@autopy x C.PyObject_Str(getptr(x_))))
pystr(x::String) = pystr_fromUTF8(x)
pystr(x::SubString{String}) = pystr_fromUTF8(x)
pystr(x::Char) = pystr(string(x))
pystr(::Type{String}, x) = (s=pystr(x); ans=pystr_asstring(s); pydel!(s); ans)
export pystr

pystr_asUTF8bytes(x::Py) = pynew(errcheck(C.PyUnicode_AsUTF8String(getptr(x))))
pystr_asUTF8vector(x::Py) = (b=pystr_asUTF8bytes(x); ans=pybytes_asvector(b); pydel!(b); ans)
pystr_asstring(x::Py) = (b=pystr_asUTF8bytes(x); ans=pybytes_asUTF8string(b); pydel!(b); ans)
