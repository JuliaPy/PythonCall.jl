"""
    module PythonCall.JlWrap

Defines the Python object wrappers around Julia objects (`juliacall.Jl` etc).
"""
module JlWrap

using ..PythonCall: PythonCall
using ..Core
using ..Core: C, Utils, pynew, @autopy, incref, decref, setptr!, getptr, pyjuliacallmodule, pycopy!, errcheck, errset, PyNULL, pyistuple, pyisnull, pyJuliaError, pydel!, pyistype, pytypecheck, pythrow, pytuple_getitem, pyisslice, pystr_asstring, pyosmodule, pyisstr
using ..Convert: pyconvert, @pyconvert, PYCONVERT_PRIORITY_WRAP, pyconvert_add_rule, pyconvert_tryconvert, pyconvertarg, pyconvert_result
using ..GC: GC

using Base: @propagate_inbounds, allocatedinline

import ..Core: Py

include("C.jl")
include("base.jl")
include("any.jl")
include("iter.jl")
include("io.jl")
include("number.jl")
include("objectarray.jl")
include("array.jl")
include("vector.jl")
include("dict.jl")
include("set.jl")
include("callback.jl")

function __init__()
    Cjl.C.with_gil() do 
        init_base()
        init_any()
        init_iter()
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
        jl.PythonCall = PythonCall
    end
end

end
