@cdef :PySlice_New PyPtr (PyPtr, PyPtr, PyPtr)

const PySlice_Type__ref = Ref(PyPtr())
PySlice_Type() = pyglobal(PySlice_Type__ref, :PySlice_Type)

PySlice_Check(o) = Py_TypeCheck(o, PySlice_Type())

PySlice_CheckExact(o) = Py_TypeCheckExact(o, PySlice_Type())

### ELLIPSIS

const Py_Ellipsis__ref = Ref(PyPtr())
Py_Ellipsis() = pyglobal(Py_Ellipsis__ref, :_Py_EllipsisObject)

PyEllipsis_Check(o) = Py_Is(o, Py_Ellipsis())

PyEllipsis_New() = (o=Py_Ellipsis(); Py_IncRef(o); o)
