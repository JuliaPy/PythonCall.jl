"""
    module PythonCall.JlWrap

Defines the Python object wrappers around Julia objects (`juliacall.AnyValue` etc).
"""
module JlWrap

using ..PythonCall
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

using Pkg: Pkg
using Base: @propagate_inbounds, allocatedinline

import ..PythonCall:
    Py,
    pyjl,
    pyjltype,
    pyisjl,
    pyjlvalue,
    pyfunc,
    pyclassmethod,
    pystaticmethod,
    pyproperty,
    pybinaryio,
    pytextio,
    pyjlraw,
    PyObjectVector,
    PyObjectMatrix,
    PyObjectArray

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
