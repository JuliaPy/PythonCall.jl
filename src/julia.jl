pyjlbasetype(::Type{T}) where {T} = checknullconvert(T, C.PyJuliaBaseValue_Type())
pyjlbasetype() = pyjlbasetype(PyObject)

pyjlrawtype(::Type{T}) where {T} = checknullconvert(T, C.PyJuliaRawValue_Type())
pyjlrawtype() = pyjlrawtype(PyObject)

"""
    pyjlraw([T=PyObject,] x)

Wrap `x` as a Python `julia.RawValue` object.
"""
pyjlraw(::Type{T}, x) where {T} = checknullconvert(T, C.PyJuliaRawValue_New(x))
pyjlraw(x) = pyjlraw(PyObject, x)
export pyjlraw

"""
    pyjl([T=PyObject,] x)

Wrap `x` as a Python `julia.AnyValue` (or subclass) object.
"""
pyjl(::Type{T}, x) where {T} = checknullconvert(T, C.PyJuliaValue_From(x))
pyjl(x) = pyjl(PyObject, x)
export pyjl

"""
    pyjlgetvalue()
"""
pyjlgetvalue(o) = pyisjl(o) ? cpyop(C.PyJuliaValue_GetValue, o) : error("Not a Julia value")
export pyjlgetvalue

"""
    pyisjl(o)

True if `o` is a `julia.ValueBase` object.
"""
pyisjl(o) = cpyop(C.PyJuliaValue_Check, o)
export pyisjl
