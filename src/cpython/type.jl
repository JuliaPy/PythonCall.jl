@cdef :PyType_IsSubtype Cint (PyPtr, PyPtr)
@cdef :PyType_Ready Cint (PyPtr,)

Py_Type(o) = GC.@preserve o UnsafePtr(Base.unsafe_convert(PyPtr, o)).type[!]
Py_TypeCheck(o, t) = PyType_IsSubtype(Py_Type(o), t) != 0
Py_TypeCheckExact(o, t) = Py_Type(o) == Base.unsafe_convert(PyPtr, t)
Py_TypeCheckFast(o, f) = PyType_IsSubtypeFast(Py_Type(o), f)

PyType_Flags(o) =
    GC.@preserve o UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).flags[]
PyType_Name(o) = GC.@preserve o unsafe_string(
    UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).name[!],
)
PyType_MRO(o) = GC.@preserve o UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).mro[!]

PyType_IsSubtypeFast(s, f) = PyType_HasFeature(s, f)
PyType_HasFeature(s, f) = !iszero(PyType_Flags(s) & f)

const PyType_Type__ref = Ref(PyPtr())
PyType_Type() = pyglobal(PyType_Type__ref, :PyType_Type)

PyType_Check(o) = Py_TypeCheck(o, Py_TPFLAGS_TYPE_SUBCLASS)

PyType_CheckExact(o) = Py_TypeCheckExact(o, PyType_Type())

PyType_FullName(o) = begin
    # get __module__
    mo = PyObject_GetAttrString(o, "__module__")
    isnull(mo) && return PYERR()
    m = PyUnicode_AsString(mo)
    Py_DecRef(mo)
    isempty(m) && PyErr_IsSet() && return PYERR()
    # get __qualname__
    no = PyObject_GetAttrString(o, "__qualname__")
    isnull(no) && return PYERR()
    n = PyUnicode_AsString(no)
    Py_DecRef(no)
    isempty(n) && PyErr_IsSet() && return PYERR()
    # done
    "$m.$n"
end

PyType_MROAsVector(o) = begin
    mro = PyType_MRO(o)
    PyPtr[PyTuple_GetItem(mro, i - 1) for i = 1:PyTuple_Size(mro)]
end
