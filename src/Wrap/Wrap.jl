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
    priority = 0
    Convert.pyconvert_add_rule_high_priority(
        "pandas:DataFrame",
        PyPandasDataFrame,
        Any,
        pyconvert_rule_pandasdataframe,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "pandas.api.extensions:ExtensionArray",
        PyList,
        Any,
        pyconvert_rule_sequence,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "collections.abc:Sequence",
        PyList,
        Any,
        pyconvert_rule_sequence,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "collections.abc:Set",
        PySet,
        Any,
        pyconvert_rule_set,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "collections.abc:Mapping",
        PyDict,
        Any,
        pyconvert_rule_mapping,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "collections.abc:Iterable",
        PyIterable,
        Any,
        pyconvert_rule_iterable,
        priority,
    )
    Convert.pyconvert_add_rule_high_priority(
        "io:IOBase",
        PyIO,
        Any,
        pyconvert_rule_io,
        priority,
    )

    pyconvert_add_rule("<arraystruct>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<arrayinterface>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<array>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<buffer>", AbstractArray, AbstractArray, pyconvert_rule_array)
    Convert.pyconvert_add_rule_high_priority(
        "<arraystruct>",
        PyArray,
        Any,
        pyconvert_rule_array_nocopy,
        1,
    )
    Convert.pyconvert_add_rule_high_priority(
        "<arrayinterface>",
        PyArray,
        Any,
        pyconvert_rule_array_nocopy,
        1,
    )
    Convert.pyconvert_add_rule_high_priority("<array>", PyArray, Any, pyconvert_rule_array_nocopy, 1)
    Convert.pyconvert_add_rule_high_priority("<buffer>", PyArray, Any, pyconvert_rule_array_nocopy, 1)
end

end
