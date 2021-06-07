PyMapping_HasKeyString(o, k) = ccall(POINTERS.PyMapping_HasKeyString, Cint, (PyPtr, Cstring), o, k)
PyMapping_SetItemString(o, k, v) = ccall(POINTERS.PyMapping_SetItemString, Cint, (PyPtr, Cstring, PyPtr), o, k, v)
PyMapping_GetItemString(o, k) = ccall(POINTERS.PyMapping_GetItemString, PyPtr, (PyPtr, Cstring), o, k)
