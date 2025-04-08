"""Interoperability with PyCall.jl"""
module PyCall

using ...PythonCall
using ...C
using ...Core

using Requires: @require

function init_pycall(PyCall::Module)
    # allow explicit conversion between PythonCall.Py and PyCall.PyObject
    # provided they are using the same interpretr
    errmsg = """
    Conversion between `PyCall.PyObject` and `PythonCall.Py` is only possible when using the same Python interpreter.

    There are two ways to achieve this:
    - Set the environment variable `JULIA_PYTHONCALL_EXE` to `"@PyCall"`. This forces PythonCall to use the same
      interpreter as PyCall, but PythonCall loses the ability to manage its own dependencies.
    - Set the environment variable `PYTHON` to `PythonCall.C.CTX.exe_path` and rebuild PyCall. This forces PyCall
      to use the same interpreter as PythonCall, but needs to be repeated whenever you switch Julia environment.
    """
    @eval function PythonCall.Py(x::$PyCall.PyObject)
        C.CTX.matches_pycall::Bool || error($errmsg)
        return pynew(C.PyPtr($PyCall.pyreturn(x)))
    end
    @eval function PyCall.PyObject(x::Py)
        C.CTX.matches_pycall::Bool || error($errmsg)
        return $PyCall.PyObject($PyCall.PyPtr(incref(getptr(x))))
    end
end

function __init__()
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)
end

end
