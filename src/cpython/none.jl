const Py_NotImplemented__ref = Ref(PyPtr())
Py_NotImplemented() = pyglobal(Py_NotImplemented__ref, :_Py_NotImplementedStruct)

PyNotImplemented_Check(o) = Py_Is(o, Py_NotImplemented())

PyNotImplemented_New() = (o=Py_NotImplemented(); Py_IncRef(o); o)

const Py_None__ref = Ref(PyPtr())
Py_None() = pyglobal(Py_None__ref, :_Py_NoneStruct)

PyNone_Check(o) = Py_Is(o, Py_None())

PyNone_New() = (o=Py_None(); Py_IncRef(o); o)

PyNone_TryConvertRule_nothing(o, ::Type{Nothing}) = putresult(nothing)
PyNone_TryConvertRule_missing(o, ::Type{Missing}) = putresult(missing)
