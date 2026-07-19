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
    pyconvert_add_rule("collections.abc:Sequence",
        PyList,
        Any,
        pyconvert_rule_sequence)
    pyconvert_add_rule("collections.abc:Set", PySet, Any, pyconvert_rule_set)
    pyconvert_add_rule("collections.abc:Mapping", PyDict, Any, pyconvert_rule_mapping)
    pyconvert_add_rule("collections.abc:Iterable",
        PyIterable,
        Any,
        pyconvert_rule_iterable)
    pyconvert_add_rule("io:IOBase", PyIO, Any, pyconvert_rule_io)
    pyconvert_add_rule(
        "pandas:DataFrame",
        PyPandasDataFrame,
        Any,
        pyconvert_rule_pandasdataframe,
    )
    pyconvert_add_rule(
        "pandas.api.extensions:ExtensionArray",
        PyList,
        Any,
        pyconvert_rule_sequence,
    )

    pyconvert_add_rule("<arraystruct>", Array, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<arrayinterface>", Array, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<array>", Array, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<buffer>", Array, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<arraystruct>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<arrayinterface>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<array>", AbstractArray, AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<buffer>", AbstractArray, AbstractArray, pyconvert_rule_array)
end

function init_wrap_rules_high_priority()
    pyconvert_add_rule_high_priority("<arraystruct>", PyArray, Any, pyconvert_rule_array_nocopy)
    pyconvert_add_rule_high_priority("<arrayinterface>", PyArray, Any, pyconvert_rule_array_nocopy)
    pyconvert_add_rule_high_priority("<array>", PyArray, Any, pyconvert_rule_array_nocopy)
    pyconvert_add_rule_high_priority("<buffer>", PyArray, Any, pyconvert_rule_array_nocopy)
end

end
