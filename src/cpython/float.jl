PyFloat_FromDouble(x) = ccall(POINTERS.PyFloat_FromDouble, PyPtr, (Cdouble,), x)
PyFloat_AsDouble(o) = ccall(POINTERS.PyFloat_AsDouble, Cdouble, (PyPtr,), o)

PyFloat_Type() = POINTERS.PyFloat_Type

PyFloat_Check(o) = Py_TypeCheck(o, PyFloat_Type())

PyFloat_CheckExact(o) = Py_TypeCheckExact(o, PyFloat_Type())

PyFloat_From(o::Union{Float16,Float32,Float64}) = PyFloat_FromDouble(o)

# "Floatable" means a 'float' or anything with a '__float__' method
PyFloatable_TryConvertRule_convert(o, ::Type{S}) where {S} = begin
    x = PyFloat_AsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    putresult(convert(S, x))
end

PyFloatable_TryConvertRule_tryconvert(o, ::Type{S}) where {S} = begin
    x = PyFloat_AsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    putresult(tryconvert(S, x))
end

# NaN is sometimes used to represent missing data of other types
PyFloatable_TryConvertRule_nothing(o, ::Type{Nothing}) = begin
    x = PyFloat_AsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    isnan(x) ? putresult(nothing) : 0
end

PyFloatable_TryConvertRule_missing(o, ::Type{Missing}) = begin
    x = PyFloat_AsDouble(o)
    ism1(x) && PyErr_IsSet() && return -1
    isnan(x) ? putresult(missing) : 0
end
