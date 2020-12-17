@cdef :PyMapping_HasKeyString Cint (PyPtr, Cstring)
@cdef :PyMapping_SetItemString Cint (PyPtr, Cstring, PyPtr)
@cdef :PyMapping_GetItemString PyPtr (PyPtr, Cstring)
