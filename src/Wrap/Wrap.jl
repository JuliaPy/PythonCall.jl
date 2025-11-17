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
using ..Convert: PyConvertRuleSpec
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

function wrap_pyconvert_rule_specs()
    return PyConvertRuleSpec[
        (func = pyconvert_rule_array_nocopy, tname = "<arraystruct>", type = PyArray, scope = Any),
        (func = pyconvert_rule_array_nocopy, tname = "<arrayinterface>", type = PyArray, scope = Any),
        (func = pyconvert_rule_array_nocopy, tname = "<array>", type = PyArray, scope = Any),
        (func = pyconvert_rule_array_nocopy, tname = "<buffer>", type = PyArray, scope = Any),
        (func = pyconvert_rule_iterable, tname = "collections.abc:Iterable", type = PyIterable, scope = PyIterable),
        (func = pyconvert_rule_sequence, tname = "collections.abc:Sequence", type = PyList, scope = PyList),
        (func = pyconvert_rule_set, tname = "collections.abc:Set", type = PySet, scope = PySet),
        (func = pyconvert_rule_mapping, tname = "collections.abc:Mapping", type = PyDict, scope = PyDict),
        (func = pyconvert_rule_io, tname = "io:IOBase", type = PyIO, scope = PyIO),
        (func = pyconvert_rule_io, tname = "_io:_IOBase", type = PyIO, scope = PyIO),
        (
            func = pyconvert_rule_pandasdataframe,
            tname = "pandas.core.frame:DataFrame",
            type = PyPandasDataFrame,
            scope = PyPandasDataFrame,
        ),
        (
            func = pyconvert_rule_sequence,
            tname = "pandas.core.arrays.base:ExtensionArray",
            type = PyList,
            scope = PyList,
        ),
        (func = pyconvert_rule_array, tname = "<arraystruct>", type = Array, scope = Array),
        (func = pyconvert_rule_array, tname = "<arrayinterface>", type = Array, scope = Array),
        (func = pyconvert_rule_array, tname = "<array>", type = Array, scope = Array),
        (func = pyconvert_rule_array, tname = "<buffer>", type = Array, scope = Array),
        (func = pyconvert_rule_array, tname = "<arraystruct>", type = AbstractArray, scope = AbstractArray),
        (func = pyconvert_rule_array, tname = "<arrayinterface>", type = AbstractArray, scope = AbstractArray),
        (func = pyconvert_rule_array, tname = "<array>", type = AbstractArray, scope = AbstractArray),
        (func = pyconvert_rule_array, tname = "<buffer>", type = AbstractArray, scope = AbstractArray),
    ]
end

end
