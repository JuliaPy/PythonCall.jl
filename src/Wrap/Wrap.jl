"""
    module PythonCall.Wrap

Defines Julia wrappers around Python objects, including `PyList`, `PyDict`, `PyArray` and `PyIO`.
"""
module Wrap

using ..PythonCall
using ..Utils
using ..NumpyDates
using ..C
using ..Core
using ..Convert
using ..PyMacro

import ..PythonCall:
    PyArray, PyDict, PyIO, PyIterable, PyList, PyPandasDataFrame, PySet, PyTable

using Base: @propagate_inbounds
using Tables: Tables
using UnsafePointers: UnsafePtr

import ..Core: Py, ispy

include("PyIterable.jl")
include("PyDict.jl")
include("PyList.jl")
include("PySet.jl")
include("PyArray.jl")
include("PyIO.jl")
include("PyTable.jl")
include("PyPandasDataFrame.jl")

function __init__()
    for source in Convert.PYCONVERT_PROTOCOLS
        pyconvert_add_rule(source, PyArray, pyconvert_rule_array_nocopy)
    end
    Convert.init_pyconvert_array_fallbacks()
    pyconvert_add_rule("collections.abc:Iterable", PyIterable, pyconvert_rule_iterable; scope=Any)
    pyconvert_add_rule("collections.abc:Sequence", PyList, pyconvert_rule_sequence)
    pyconvert_add_rule("collections.abc:Set", PySet, pyconvert_rule_set)
    pyconvert_add_rule("collections.abc:Mapping", PyDict, pyconvert_rule_mapping)
    pyconvert_add_rule("io:IOBase", PyIO, pyconvert_rule_io; scope=Any)
    # This private C base is distinct from the public io.IOBase ABC.
    pyconvert_add_rule("_io:_IOBase", PyIO, pyconvert_rule_io; scope=Any)
    pyconvert_add_rule("pandas:DataFrame", PyPandasDataFrame, pyconvert_rule_pandasdataframe; scope=Any)
    pyconvert_add_rule("pandas.api.extensions:ExtensionArray", PyList, pyconvert_rule_sequence; scope=Any)
end

end
