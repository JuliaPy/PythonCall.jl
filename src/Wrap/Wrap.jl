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

end
