const PYWHOME = Base.cconvert(Cwstring, PYHOME)
const PYWPROGNAME = Base.cconvert(Cwstring, PYPROGNAME)
const PYISSTACKLESS = false

function __init__()
    dlopen(PYLIB)
    cpycall_voidx(Val(:Py_SetPythonHome), pointer(PYWHOME))
    cpycall_voidx(Val(:Py_SetProgramName), pointer(PYWPROGNAME))
    cpycall_voidx(Val(:Py_Initialize))
end
