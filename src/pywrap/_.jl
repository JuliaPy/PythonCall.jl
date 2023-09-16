"""
    module _pywrap

Defines Julia wrappers around Python objects, including `PyList`, `PyDict`, `PyArray` and `PyIO`.
"""
module _pywrap

using .._Utils: _Utils as Utils
using .._CPython: _CPython as C
using .._Py
using .._Py: @autopy, unsafe_pynext, pyisnull, PyNULL, getptr, pydel!
using .._pyconvert: pyconvert, pyconvert_tryconvert, pyconvert_unconverted, pyconvert_isunconverted, pyconvert_return, pyconvert_result
using .._pymacro

using Base: @propagate_inbounds
using Tables: Tables

import .._Py: Py, ispy
import .._pyconvert: pyconvert_add_rule, PYCONVERT_PRIORITY_ARRAY, PYCONVERT_PRIORITY_CANONICAL, PYCONVERT_PRIORITY_NORMAL

include("PyIterable.jl")
include("PyDict.jl")
include("PyList.jl")
include("PySet.jl")
include("PyArray.jl")
include("PyIO.jl")
include("PyTable.jl")
include("PyPandasDataFrame.jl")

function __init__()
    priority = PYCONVERT_PRIORITY_ARRAY
    pyconvert_add_rule("<arraystruct>", PyArray, pyconvert_rule_array_nocopy, priority)
    pyconvert_add_rule("<arrayinterface>", PyArray, pyconvert_rule_array_nocopy, priority)
    pyconvert_add_rule("<array>", PyArray, pyconvert_rule_array_nocopy, priority)
    pyconvert_add_rule("<buffer>", PyArray, pyconvert_rule_array_nocopy, priority)

    priority = PYCONVERT_PRIORITY_CANONICAL
    pyconvert_add_rule("collections.abc:Iterable", PyIterable, pyconvert_rule_iterable, priority)
    pyconvert_add_rule("collections.abc:Sequence", PyList, pyconvert_rule_sequence, priority)
    pyconvert_add_rule("collections.abc:Set", PySet, pyconvert_rule_set, priority)
    pyconvert_add_rule("collections.abc:Mapping", PyDict, pyconvert_rule_mapping, priority)
    pyconvert_add_rule("io:IOBase", PyIO, pyconvert_rule_io, priority)
    pyconvert_add_rule("_io:_IOBase", PyIO, pyconvert_rule_io, priority)
    pyconvert_add_rule("pandas.core.frame:DataFrame", PyPandasDataFrame, pyconvert_rule_pandasdataframe, priority)
    pyconvert_add_rule("pandas.core.arrays.base:ExtensionArray", PyList, pyconvert_rule_sequence, priority)

    priority = PYCONVERT_PRIORITY_NORMAL
    pyconvert_add_rule("<arraystruct>", Array, pyconvert_rule_array, priority)
    pyconvert_add_rule("<arrayinterface>", Array, pyconvert_rule_array, priority)
    pyconvert_add_rule("<array>", Array, pyconvert_rule_array, priority)
    pyconvert_add_rule("<buffer>", Array, pyconvert_rule_array, priority)
    pyconvert_add_rule("<arraystruct>", AbstractArray, pyconvert_rule_array, priority)
    pyconvert_add_rule("<arrayinterface>", AbstractArray, pyconvert_rule_array, priority)
    pyconvert_add_rule("<array>", AbstractArray, pyconvert_rule_array, priority)
    pyconvert_add_rule("<buffer>", AbstractArray, pyconvert_rule_array, priority)
end

end
