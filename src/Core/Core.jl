module Core

using ..PythonCall
using ..C: C
using ..GC: GC
using ..Utils: Utils
using Base: @propagate_inbounds, @kwdef
using Dates:
    Date,
    Time,
    DateTime,
    year,
    month,
    day,
    hour,
    minute,
    second,
    millisecond,
    microsecond,
    nanosecond
using MacroTools: @capture
using Markdown: Markdown

import PythonCall:
    VERSION,
    Py,
    PyException,
    ispy,
    pyis,
    pyrepr,
    pyascii,
    pyhasattr,
    pygetattr,
    pysetattr,
    pydelattr,
    pyissubclass,
    pyisinstance,
    pyhash,
    pytruth,
    pynot,
    pylen,
    pyhasitem,
    pygetitem,
    pysetitem,
    pydelitem,
    pydir,
    pycall,
    pyeq,
    pyne,
    pyle,
    pylt,
    pyge,
    pygt,
    pycontains,
    pyin,
    pyneg,
    pypos,
    pyabs,
    pyinv,
    pyindex,
    pyadd,
    pysub,
    pymul,
    pymatmul,
    pyfloordiv,
    pytruediv,
    pymod,
    pydivmod,
    pylshift,
    pyrshift,
    pyand,
    pyxor,
    pyor,
    pyiadd,
    pyisub,
    pyimul,
    pyimatmul,
    pyifloordiv,
    pyitruediv,
    pyimod,
    pyilshift,
    pyirshift,
    pyiand,
    pyixor,
    pyior,
    pypow,
    pyipow,
    pyiter,
    pynext,
    pybool,
    pystr,
    pybytes,
    pyint,
    pyfloat,
    pycomplex,
    pytype,
    pyslice,
    pyrange,
    pytuple,
    pylist,
    pycollist,
    pyrowlist,
    pyset,
    pyfrozenset,
    pydict,
    pydate,
    pytime,
    pydatetime,
    pyfraction,
    pyeval,
    pyexec,
    @pyeval,
    @pyexec,
    pywith,
    pyimport,
    pyprint,
    pyhelp,
    pyall,
    pyany,
    pycallable,
    pycompile,
    @pyconst,
    pyconvert,
    pynew,
    pyisnull,
    pycopy!,
    getptr,
    pydel!,
    unsafe_pynext

const ROOT_DIR = dirname(dirname(@__DIR__))

include("Py.jl")
include("err.jl")
include("consts.jl")
include("builtins.jl")
include("stdlib.jl")
include("juliacall.jl")
include("pyconst_macro.jl")

function __init__()
    init_consts()
    init_datetime()
    init_stdlib()
    init_juliacall()
end

end
