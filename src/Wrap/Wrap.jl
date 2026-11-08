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
using ..Convert: pyconvert_return
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
    # Native materialized containers precede explicit wrappers on the same source.
    pyconvert_add_rule("builtins:list", Vector, Convert.pyconvert_rule_iterable; scope=Any)
    pyconvert_add_rule("builtins:dict", Dict, Convert.pyconvert_rule_mapping; scope=Any)
    for source in ("builtins:set", "builtins:frozenset")
        pyconvert_add_rule(source, Set, Convert.pyconvert_rule_iterable; scope=Any)
    end
    # Definition order retains struct/interface/array/buffer precedence. Protocols
    # are independent matchers; no protocol implies another protocol.
    for source in ("<arraystruct>", "<arrayinterface>", "<array>", "<buffer>")
        pyconvert_add_rule(source, Array, pyconvert_rule_array; scope=Any)
        pyconvert_add_rule(source, PyArray, pyconvert_rule_array_nocopy)
        pyconvert_add_rule(source, AbstractArray, pyconvert_rule_array)
    end
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
