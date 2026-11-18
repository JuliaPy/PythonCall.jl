ispy(::PyString) = true
Py(x::PyString) = x.py

Base.ncodeunits(x::PyString) = x.nbytes
Base.codeunit(::PyString) = UInt8
Base.codeunit(::Type{PyString}) = UInt8
Base.codeunit(x::PyString, i::Integer) = @inbounds unsafe_load(x.ptr + (i - 1))

Base.isvalid(x::PyString, i::Int) =
    Utils.utf8_isvalid(j -> @inbounds(unsafe_load(x.ptr + (j - 1))), x.nbytes, i)

Base.iterate(x::PyString, i::Int = 1) =
    Utils.utf8_iterate(x, j -> @inbounds(unsafe_load(x.ptr + (j - 1))), x.nbytes, i)

Base.String(x::PyString) = unsafe_string(x.ptr, x.nbytes)
