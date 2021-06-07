PySlice_New(x, y, z) = ccall(POINTERS.PySlice_New, PyPtr, (PyPtr, PyPtr, PyPtr), x, y, z)

PySlice_Type() = POINTERS.PySlice_Type

PySlice_Check(o) = Py_TypeCheck(o, PySlice_Type())

PySlice_CheckExact(o) = Py_TypeCheckExact(o, PySlice_Type())

### ELLIPSIS

Py_Ellipsis() = POINTERS._Py_EllipsisObject

PyEllipsis_Check(o) = Py_Is(o, Py_Ellipsis())

PyEllipsis_New() = (o = Py_Ellipsis(); Py_IncRef(o); o)
