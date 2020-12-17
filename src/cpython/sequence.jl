@cdef :PySequence_Length Py_ssize_t (PyPtr,)
@cdef :PySequence_GetItem PyPtr (PyPtr, Py_ssize_t)
@cdef :PySequence_SetItem Cint (PyPtr, Py_ssize_t, PyPtr)
@cdef :PySequence_Contains Cint (PyPtr, PyPtr)
