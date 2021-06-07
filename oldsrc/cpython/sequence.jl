PySequence_Length(o) = ccall(POINTERS.PySequence_Length, Py_ssize_t, (PyPtr,), o)
PySequence_GetItem(o, k) = ccall(POINTERS.PySequence_GetItem, PyPtr, (PyPtr, Py_ssize_t), o, k)
PySequence_SetItem(o, k, v) = ccall(POINTERS.PySequence_SetItem, Cint, (PyPtr, Py_ssize_t, PyPtr), o, k, v)
PySequence_Contains(o, v) = ccall(POINTERS.PySequence_Contains, Cint, (PyPtr, PyPtr), o, v)
