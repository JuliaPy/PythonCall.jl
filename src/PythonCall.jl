module PythonCall

using Base: @propagate_inbounds
using MacroTools, Dates, Tables, Markdown

include("utils.jl")

include("cpython/CPython.jl")

include("Py.jl")
include("err.jl")
include("config.jl")
include("convert.jl")
# abstract interfaces
include("abstract/object.jl")
include("abstract/iter.jl")
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
include("concrete/method.jl")
include("concrete/datetime.jl")
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
include("pywrap/PyIO.jl")
include("pywrap/PyTable.jl")
include("pywrap/PyPandasDataFrame.jl")
# misc
include("with.jl")
include("multimedia.jl")
include("pyconst_macro.jl")
include("juliacall.jl")

function __init__()
    C.with_gil() do
        init_consts()
        init_pyconvert()
        init_datetime()
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
    end
end

end
