"""
    module PythonCall.JlWrap

Defines the Python object wrappers around Julia objects (`juliacall.AnyValue` etc).
"""
module JlWrap

using ..PythonCall
using ..Utils
using ..C
using ..Core
using ..Convert
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
    pybinaryio,
    pytextio

using Pkg: Pkg
using Base: @propagate_inbounds, allocatedinline

import ..Core: Py

include("C.jl")
include("base.jl")
include("raw.jl")
include("any.jl")
include("iter.jl")
include("type.jl")
include("module.jl")
include("io.jl")
include("number.jl")
include("objectarray.jl")
include("array.jl")
include("vector.jl")
include("dict.jl")
include("set.jl")
include("callback.jl")

function __init__()
    init_base()
    init_raw()
    init_any()
    init_iter()
    init_type()
    init_module()
    init_io()
    init_number()
    init_array()
    init_vector()
    init_dict()
    init_set()
    init_callback()
    # add packages to juliacall
    jl = pyjuliacallmodule
    jl.Core = Base.Core
    jl.Base = Base
    jl.Main = Main
    jl.Pkg = Pkg
    jl.PythonCall = PythonCall
end

end
