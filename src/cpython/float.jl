@cdef :PyFloat_FromDouble PyPtr (Cdouble,)
@cdef :PyFloat_AsDouble Cdouble (PyPtr,)

const PyFloat_Type__ref = Ref(PyPtr())
PyFloat_Type() = pyglobal(PyFloat_Type__ref, :PyFloat_Type)

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
