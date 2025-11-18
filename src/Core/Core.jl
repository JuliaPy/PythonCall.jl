"""
    module PythonCall.Core

Defines the `Py` type and directly related functions.
"""
module Core

const VERSION = v"0.9.30"
const ROOT_DIR = dirname(dirname(@__DIR__))

using ..PythonCall
using ..C
using ..GC: GC
using ..Utils

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
using MacroTools: MacroTools, @capture
using Markdown: Markdown

import ..PythonCall:
    @pyconst,
    @pyeval,
    @pyexec,
    getptr,
    ispy,
    Py,
    pyabs,
    pyadd,
    pyall,
    pyand,
    pyany,
    pyascii,
    pybool,
    pybuiltins,
    pybytes,
    pycall,
    pycallable,
    pycollist,
    pycompile,
    pycomplex,
    pycontains,
    pyconvert,
    pycopy!,
    pydate,
    pydatetime,
    pydel!,
    pydelattr,
    pydelitem,
    pydict,
    pydir,
    pydivmod,
    pyeq,
    pyeval,
    PyException,
    pyexec,
    pyfloat,
    pyfloordiv,
    pyfraction,
    pyfrozenset,
    pyge,
    pygetattr,
    pygetitem,
    pygt,
    pyhasattr,
    pyhash,
    pyhasitem,
    pyhelp,
    pyiadd,
    pyiand,
    pyifloordiv,
    pyilshift,
    pyimatmul,
    pyimod,
    pyimport,
    pyimul,
    pyin,
    pyindex,
    pyint,
    pyinv,
    pyior,
    pyipow,
    pyirshift,
    pyis,
    pyisinstance,
    pyisnull,
    pyissubclass,
    pyisub,
    pyiter,
    pyitruediv,
    pyixor,
    pyle,
    pylen,
    pylist,
    pylshift,
    pylt,
    pymatmul,
    pymod,
    pymul,
    pyne,
    pyneg,
    pynew,
    pynext,
    pynot,
    pyor,
    pypos,
    pypow,
    pyprint,
    pyrange,
    pyrepr,
    pyrowlist,
    pyrshift,
    pyset,
    pysetattr,
    pysetitem,
    pyslice,
    pystr,
    pysub,
    pytime,
    pytruediv,
    pytruth,
    pytuple,
    pytype,
    pywith,
    pyxor,
    unsafe_pynext
    
export
    _base_datetime,
    _base_pydatetime,
    @autopy,
    BUILTINS,
    decref,
    errcheck,
    errclear,
    errmatches,
    errset,
    getptr,
    incref,
    iserrset_ambig,
    pybool_asbool,
    pybytes_asUTF8string,
    pybytes_asvector,
    pycallargs,
    pycomplex_ascomplex,
    pycopy!,
    pydatetime_isaware,
    pydatetimetype,
    pydel!,
    pydict_setitem,
    pyfloat_asdouble,
    pyisbytes,
    pyiscomplex,
    pyisFalse,
    pyisfloat,
    pyisint,
    pyisnone,
    pyisnot,
    pyisnull,
    pyisrange,
    pyisslice,
    pyisstr,
    pyisTrue,
    pyistuple,
    pyistype,
    pyjuliacallmodule,
    pyJuliaError,
    pylist_setitem,
    pymodulehooks,
    pynew,
    pynotin,
    PyNULL,
    pynulllist,
    pynulltuple,
    pyosmodule,
    pyset_add,
    pystr_asstring,
    pystr_asUTF8vector,
    pystr_fromUTF8,
    pystr_intern!,
    pysysmodule,
    pythrow,
    pytime_isaware,
    pytuple_getitem,
    pytuple_setitem,
    pytypecheck,
    setptr!,
    unsafe_pynext

include("Py.jl")
include("err.jl")
include("config.jl")
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
