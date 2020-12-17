@cdef :PyComplex_FromDoubles PyPtr (Cdouble, Cdouble)
@cdef :PyComplex_RealAsDouble Cdouble (PyPtr,)
@cdef :PyComplex_ImagAsDouble Cdouble (PyPtr,)

const PyComplex_Type__ref = Ref(PyPtr())
PyComplex_Type() = pyglobal(PyComplex_Type__ref, :PyComplex_Type)

PyComplex_Check(o) = Py_TypeCheck(o, PyComplex_Type())

PyComplex_CheckExact(o) = Py_TypeCheckExact(o, PyComplex_Type())

PyComplex_From(x::Union{Float16,Float32,Float64}) = PyComplex_FromDoubles(x, 0)
PyComplex_From(x::Complex{<:Union{Float16,Float32,Float64}}) = PyComplex_FromDoubles(real(x), imag(x))

PyComplex_TryConvertRule_convert(o, ::Type{T}, ::Type{S}) where {T,S} = begin
    x = PyComplex_RealAsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    y = PyComplex_ImagAsDouble(o)
    ism1(y) && PyErr_IsSet() && return -1
    z = Complex(x, y)
    putresult(T, convert(S, z))
end

PyComplex_TryConvertRule_tryconvert(o, ::Type{T}, ::Type{S}) where {T,S} = begin
    x = PyComplex_RealAsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    y = PyComplex_ImagAsDouble(o)
    ism1(y) && PyErr_IsSet() && return -1
    z = Complex(x, y)
    putresult(T, tryconvert(S, z))
end
