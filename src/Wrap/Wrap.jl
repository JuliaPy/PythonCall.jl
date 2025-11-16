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
    pyconvert_add_rule(pyconvert_rule_array_nocopy, "<arraystruct>", PyArray, Any)
    pyconvert_add_rule(pyconvert_rule_array_nocopy, "<arrayinterface>", PyArray, Any)
    pyconvert_add_rule(pyconvert_rule_array_nocopy, "<array>", PyArray, Any)
    pyconvert_add_rule(pyconvert_rule_array_nocopy, "<buffer>", PyArray, Any)

    pyconvert_add_rule(
        pyconvert_rule_iterable,
        "collections.abc:Iterable",
        PyIterable,
        Any,
    )
    pyconvert_add_rule(
        pyconvert_rule_sequence,
        "collections.abc:Sequence",
        PyList,
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_set, "collections.abc:Set", PySet, Any)
    pyconvert_add_rule(pyconvert_rule_mapping, "collections.abc:Mapping", PyDict, Any)
    pyconvert_add_rule(pyconvert_rule_io, "io:IOBase", PyIO, Any)
    pyconvert_add_rule(pyconvert_rule_io, "_io:_IOBase", PyIO, Any)
    pyconvert_add_rule(
        pyconvert_rule_pandasdataframe,
        "pandas.core.frame:DataFrame",
        PyPandasDataFrame,
        Any,
    )
    pyconvert_add_rule(
        pyconvert_rule_sequence,
        "pandas.core.arrays.base:ExtensionArray",
        PyList,
        Any,
    )

    pyconvert_add_rule(pyconvert_rule_array, "<arraystruct>", Array)
    pyconvert_add_rule(pyconvert_rule_array, "<arrayinterface>", Array)
    pyconvert_add_rule(pyconvert_rule_array, "<array>", Array)
    pyconvert_add_rule(pyconvert_rule_array, "<buffer>", Array)
    pyconvert_add_rule(pyconvert_rule_array, "<arraystruct>", AbstractArray)
    pyconvert_add_rule(pyconvert_rule_array, "<arrayinterface>", AbstractArray)
    pyconvert_add_rule(pyconvert_rule_array, "<array>", AbstractArray)
    pyconvert_add_rule(pyconvert_rule_array, "<buffer>", AbstractArray)
end

end
