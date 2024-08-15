"""
    module PythonCall.JlWrap

Defines the Python object wrappers around Julia objects (`juliacall.Jl` etc).
"""
module JlWrap

using ..PythonCall: PythonCall
using ..Core
using ..Core:
    C,
    Utils,
    pynew,
    @autopy,
    incref,
    decref,
    setptr!,
    getptr,
    pyjuliacallmodule,
    pycopy!,
    errcheck,
    errset,
    PyNULL,
    pyistuple,
    pyisnull,
    pyJuliaError,
    pydel!,
    pyistype,
    pytypecheck,
    pythrow,
    pytuple_getitem,
    pyisslice,
    pystr_asstring,
    pyosmodule,
    pyisstr
using ..Convert:
    pyconvert,
    @pyconvert,
    PYCONVERT_PRIORITY_WRAP,
    pyconvert_add_rule,
    pyconvert_tryconvert,
    pyconvertarg,
    pyconvert_result
using ..GC: GC
using ..GIL: GIL

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
