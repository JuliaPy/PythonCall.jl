module PyCallExt

using PythonCall
using PythonCall.Core
using PythonCall.C

using PyCall: PyCall

import PythonCall: Py

# true if PyCall and PythonCall are using the same interpreter
const SAME = Ref{Bool}(false)

function __init__()
    # see if PyCall and PythonCall are using the same interpreter by checking if a couple of memory addresses are the same
    ptr1 = C.Py_GetVersion()
    ptr2 = ccall(PyCall.@pysym(:Py_GetVersion), Ptr{Cchar}, ())
    SAME[] = ptr1 == ptr2
    if PythonCall.C.CTX.which == :PyCall
        @assert SAME[]
    end
end

# allow explicit conversion between PythonCall.Py and PyCall.PyObject
# provided they are using the same interpretr
const ERRMSG = """
Conversion between `PyCall.PyObject` and `PythonCall.Py` is only possible when using the same Python interpreter.

There are two ways to achieve this:
- Set the environment variable `JULIA_PYTHONCALL_EXE` to `"@PyCall"`. This forces PythonCall to use the same
  interpreter as PyCall, but PythonCall loses the ability to manage its own dependencies.
- Set the environment variable `PYTHON` to `PythonCall.python_executable_path()` and rebuild PyCall. This forces
  PyCall to use the same interpreter as PythonCall, but needs to be repeated whenever you switch Julia environment.
"""

function Py(x::PyCall.PyObject)
    SAME[] || error(ERRMSG)
    return pynew(C.PyPtr(PyCall.pyreturn(x)))
end

function PyCall.PyObject(x::Py)
    SAME[] || error(ERRMSG)
    return PyCall.PyObject(PyCall.PyPtr(getptr(incref(x))))
end

end
