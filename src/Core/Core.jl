"""
    module PythonCall.Core

Defines the `Py` type and directly related functions.
"""
module Core

const VERSION = v"0.9.18"
const ROOT_DIR = dirname(dirname(@__DIR__))

using ..PythonCall: PythonCall  # needed for docstring cross-refs
using ..C: C
using ..GC: GC
using ..Utils: Utils
using Base: @propagate_inbounds, @kwdef
using Dates: Date, Time, DateTime, year, month, day, hour, minute, second, millisecond, microsecond, nanosecond
using MacroTools: MacroTools, @capture
using Markdown: Markdown

include("Py.jl")
include("err.jl")
include("config.jl")
include("consts.jl")
include("builtins.jl")
include("stdlib.jl")
include("juliacall.jl")
include("pyconst_macro.jl")

function __init__()
    C.with_gil() do
        init_consts()
        init_datetime()
        init_stdlib()
        init_juliacall()
    end
end

end
