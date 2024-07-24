"""
    module PythonCall.Compat

Misc bits and bobs for compatibility.
"""
module Compat
using ..PythonCall: PythonCall  # needed for docstring cross-refs
using ..Core
using ..Core:
    Core,
    C,
    Utils,
    pynew,
    incref,
    getptr,
    pycopy!,
    pymodulehooks,
    pyisnull,
    pybytes_asvector,
    pysysmodule,
    pyosmodule,
    pystr_fromUTF8
using ..Convert: pyconvert, @pyconvert
using ..Wrap: PyArray, PyPandasDataFrame
using Serialization: Serialization, AbstractSerializer, serialize, deserialize
using Tables: Tables
using Requires: @require

include("gui.jl")
include("ipython.jl")
include("multimedia.jl")
include("serialization.jl")
include("tables.jl")
include("pycall.jl")

function __init__()
    init_gui()
    init_pyshow()
    init_tables()
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)
end
end
