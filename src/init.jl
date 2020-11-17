const PYWHOME = Base.cconvert(Cwstring, PYHOME)
const PYWPROGNAME = Base.cconvert(Cwstring, PYPROGNAME)
const PYISSTACKLESS = false

function __init__()
    C.Py_SetPythonHome(pointer(PYWHOME))
    C.Py_SetProgramName(pointer(PYWPROGNAME))
    C.Py_Initialize()
end
