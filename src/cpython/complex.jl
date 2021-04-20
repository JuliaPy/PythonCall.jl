PyComplex_FromDoubles(re, im) = ccall(POINTERS.PyComplex_FromDoubles, PyPtr, (Cdouble, Cdouble), re, im)
PyComplex_RealAsDouble(o) = ccall(POINTERS.PyComplex_RealAsDouble, Cdouble, (PyPtr,), o)
PyComplex_ImagAsDouble(o) = ccall(POINTERS.PyComplex_ImagAsDouble, Cdouble, (PyPtr,), o)
PyComplex_AsCComplex(o) = ccall(POINTERS.PyComplex_AsCComplex, Py_complex, (PyPtr,), o)

PyComplex_Type() = POINTERS.PyComplex_Type

PyComplex_Check(o) = Py_TypeCheck(o, PyComplex_Type())

PyComplex_CheckExact(o) = Py_TypeCheckExact(o, PyComplex_Type())

PyComplex_AsComplex(o) = begin
    r = PyComplex_AsCComplex(o)
    Complex(r.real, r.imag)
end

PyComplex_From(x::Union{Float16,Float32,Float64}) = PyComplex_FromDoubles(x, 0)
PyComplex_From(x::Complex{<:Union{Float16,Float32,Float64}}) =
    PyComplex_FromDoubles(real(x), imag(x))

# "Complexable" means a 'complex' or anything with a '__complex__' method
PyComplexable_TryConvertRule_convert(o, ::Type{S}) where {S} = begin
    x = PyComplex_AsComplex(o)
    ism1(x) && PyErr_IsSet() && return -1
    putresult(convert(S, x))
end

PyComplexable_TryConvertRule_tryconvert(o, ::Type{S}) where {S} = begin
    x = PyComplex_AsComplex(o)
    ism1(x) && PyErr_IsSet() && return -1
    putresult(tryconvert(S, x))
end
