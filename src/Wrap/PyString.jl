function PyString(x)
    py = ispy(x) ? Py(x) : pystr(x)
    n = Ref{C.Py_ssize_t}()
    ptr = C.PyUnicode_AsUTF8AndSize(py, n)
    ptr == C_NULL && pythrow()
    return PyString(Val(:new), py, Ptr{UInt8}(ptr), Int(n[]))
end

ispy(::PyString) = true
Py(x::PyString) = x.py

pyconvert_rule_string(::Type{PyString}, x::Py) = pyconvert_return(PyString(x))

Base.ncodeunits(x::PyString) = x.length
Base.codeunit(::PyString) = UInt8
Base.@propagate_inbounds function Base.codeunit(x::PyString, i::Integer)
    @boundscheck checkbounds(1:x.length, i)
    return unsafe_load(x.ptr, i)
end
Base.isvalid(x::PyString, i::Int) = Utils.utf8_isvalid(x, x.length, i)
Base.iterate(x::PyString, i::Int = 1) = Utils.utf8_iterate(x, x.length, i)
