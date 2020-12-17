const Py_None__ref = Ref(PyPtr())
Py_None() = pyglobal(Py_None__ref, :_Py_NoneStruct)

PyNone_Check(o) = Py_Is(o, Py_None())

PyNone_New() = (o=Py_None(); Py_IncRef(o); o)

PyNone_TryConvertRule_nothing(o, ::Type{T}, ::Type{Nothing}) where {T} = putresult(T, nothing)
PyNone_TryConvertRule_missing(o, ::Type{T}, ::Type{Missing}) where {T} = putresult(T, missing)
