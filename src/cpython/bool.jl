const PyBool_Type__ref = Ref(PyPtr())
PyBool_Type() = pyglobal(PyBool_Type__ref, :PyBool_Type)

const Py_True__ref = Ref(PyPtr())
Py_True() = pyglobal(Py_True__ref, :_Py_TrueStruct)

const Py_False__ref = Ref(PyPtr())
Py_False() = pyglobal(Py_False__ref, :_Py_FalseStruct)

PyBool_From(x::Bool) = (o = x ? Py_True() : Py_False(); Py_IncRef(o); o)

PyBool_Check(o) = Py_TypeCheckExact(o, PyBool_Type())

PyBool_TryConvertRule_bool(o, ::Type{Bool}) =
    if Py_Is(o, Py_True())
        putresult(true)
    elseif Py_Is(o, Py_False())
        putresult(false)
    else
        PyErr_SetString(PyExc_TypeError(), "Expecting a 'bool' but got a '$(PyType_Name(Py_Type(o)))'")
        -1
    end
