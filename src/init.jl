function __init__()
    PYLIBPTR[] = dlopen(PYLIBPATH)
    cpycall_voidx(Val(:Py_SetPythonHome), pointer(PYWHOME))
    cpycall_voidx(Val(:Py_Initialize))
end
