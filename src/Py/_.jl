"""
    module _Py

Defines the `Py` type and directly related functions.
"""
module _Py

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(dirname(@__DIR__))

using .._CPython: _CPython as C
using .._Utils: _Utils as Utils
using Base: @propagate_inbounds, @kwdef
using Dates: Date, Time, DateTime, year, month, day, hour, minute, second, millisecond, microsecond, nanosecond
using MacroTools: @capture
using Markdown: Markdown
# using MacroTools, Dates, Tables, Markdown, Serialization, Requires, Pkg, REPL

include("gc.jl")
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
