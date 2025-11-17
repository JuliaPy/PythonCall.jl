"""
    module PythonCall.JlWrap

Defines the Python object wrappers around Julia objects (`juliacall.Jl` etc).
"""
module JlWrap

using ..PythonCall
using ..Utils
using ..NumpyDates: NumpyDates
using ..C
using ..Core
using ..Convert
using ..Convert: PyConvertRuleSpec
using ..GC: GC
using ..GIL

import ..PythonCall:
    pyfunc,
    pyclassmethod,
    pystaticmethod,
    pyproperty,
    pyjl,
    pyjltype,
    pyisjl,
    pyjlvalue,
    pyjlraw,
    pyjlcollection,
    pyjlarray,
    pyjldict,
    pyjlset,
    pybinaryio,
    pytextio,
    PyObjectVector,
    PyObjectMatrix,
    PyObjectArray

using Base: @propagate_inbounds, allocatedinline

import ..Core: Py

include("C.jl")
include("base.jl")
include("any.jl")
include("io.jl")
include("objectarray.jl")
include("collection.jl")
include("array.jl")
include("vector.jl")
include("dict.jl")
include("set.jl")
include("callback.jl")

function __init__()
    init_base()
    init_any()
    init_io()
    init_collection()
    init_array()
    init_vector()
    init_dict()
    init_set()
    # add packages to juliacall
    jl = pyjuliacallmodule
    jl.Core = Base.Core
    jl.Base = Base
    jl.Main = Main
    jl.PythonCall = PythonCall
end

end
