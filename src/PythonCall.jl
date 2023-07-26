module PythonCall

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(@__DIR__)

using Base: @propagate_inbounds
using MacroTools, Dates, Tables, Markdown, Serialization, Requires, Pkg, REPL

include("utils.jl")

include("cpython/CPython.jl")

include("gc.jl")
include("Py.jl")
include("err.jl")
include("config.jl")
include("convert.jl")
# abstract interfaces
include("abstract/object.jl")
include("abstract/iter.jl")
include("abstract/builtins.jl")
include("abstract/number.jl")
include("abstract/collection.jl")
# concrete types
include("concrete/import.jl")
include("concrete/consts.jl")
include("concrete/str.jl")
include("concrete/bytes.jl")
include("concrete/tuple.jl")
include("concrete/list.jl")
include("concrete/dict.jl")
include("concrete/bool.jl")
include("concrete/int.jl")
include("concrete/float.jl")
include("concrete/complex.jl")
include("concrete/set.jl")
include("concrete/slice.jl")
include("concrete/range.jl")
include("concrete/none.jl")
include("concrete/type.jl")
include("concrete/fraction.jl")
include("concrete/datetime.jl")
include("concrete/code.jl")
include("concrete/ctypes.jl")
include("concrete/numpy.jl")
include("concrete/pandas.jl")
# @py
# anything below can depend on @py, anything above cannot
include("py_macro.jl")
# jlwrap
include("jlwrap/base.jl")
include("jlwrap/raw.jl")
include("jlwrap/callback.jl")
include("jlwrap/any.jl")
include("jlwrap/module.jl")
include("jlwrap/type.jl")
include("jlwrap/iter.jl")
include("jlwrap/objectarray.jl")
include("jlwrap/array.jl")
include("jlwrap/vector.jl")
include("jlwrap/dict.jl")
include("jlwrap/set.jl")
include("jlwrap/number.jl")
include("jlwrap/io.jl")
# pywrap
include("pywrap/PyIterable.jl")
include("pywrap/PyList.jl")
include("pywrap/PySet.jl")
include("pywrap/PyDict.jl")
include("pywrap/PyArray.jl")
include("pywrap/PyIO.jl")
include("pywrap/PyTable.jl")
include("pywrap/PyPandasDataFrame.jl")
# misc
include("pyconst_macro.jl")
include("juliacall.jl")
include("compat/stdlib.jl")
include("compat/with.jl")
include("compat/multimedia.jl")
include("compat/serialization.jl")
include("compat/gui.jl")
include("compat/ipython.jl")
include("compat/tables.jl")

function __init__()
    C.with_gil() do
        init_consts()
        init_pyconvert()
        init_datetime()
        # juliacall/jlwrap
        init_juliacall()
        init_jlwrap_base()
        init_jlwrap_raw()
        init_jlwrap_callback()
        init_jlwrap_any()
        init_jlwrap_module()
        init_jlwrap_type()
        init_jlwrap_iter()
        init_jlwrap_array()
        init_jlwrap_vector()
        init_jlwrap_dict()
        init_jlwrap_set()
        init_jlwrap_number()
        init_jlwrap_io()
        init_juliacall_2()
        # compat
        init_stdlib()
        init_pyshow()
        init_gui()
        init_tables()
        init_ctypes()
        init_numpy()
        init_pandas()
    end
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)
end

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
    @eval function Py(x::$PyCall.PyObject)
        C.CTX.matches_pycall::Bool || error($errmsg)
        return pynew(C.PyPtr($PyCall.pyreturn(x)))
    end
    @eval function $PyCall.PyObject(x::Py)
        C.CTX.matches_pycall::Bool || error($errmsg)
        return $PyCall.PyObject($PyCall.PyPtr(incref(getptr(x))))
    end
end

end
