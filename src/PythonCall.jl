module PythonCall

const VERSION = v"0.9.15"
const ROOT_DIR = dirname(@__DIR__)

include("utils/_.jl")
include("CPython/_.jl")
include("Py/_.jl")
include("pyconvert/_.jl")
include("pymacro/_.jl")
include("pywrap/_.jl")
include("jlwrap/_.jl")
include("compat/_.jl")

# re-export everything
for m in [:_Py, :_pyconvert, :_pymacro, :_pywrap, :_jlwrap, :_compat]
    for k in names(@eval($m))
        if k != m
            @eval using .$m: $k
            @eval export $k
        end
    end
end

# non-exported API
for k in [:C, :GC, :pynew, :pyisnull, :pycopy!, :getptr, :pydel!, :unsafe_pynext, :PyNULL, :CONFIG]
    @eval const $k = _Py.$k
end
for k in [:pyconvert_add_rule, :pyconvert_return, :pyconvert_unconverted, :PYCONVERT_PRIORITY_WRAP, :PYCONVERT_PRIORITY_ARRAY, :PYCONVERT_PRIORITY_CANONICAL, :PYCONVERT_PRIORITY_NORMAL, :PYCONVERT_PRIORITY_FALLBACK]
    @eval const $k = _pyconvert.$k
end
for k in [:event_loop_on, :event_loop_off, :fix_qt_plugin_path]
    @eval const $k = _compat.$k
end

# not API but used in tests
for k in [:pyjlanytype, :pyjlarraytype, :pyjlvectortype, :pyjlbinaryiotype, :pyjltextiotype, :pyjldicttype, :pyjlmoduletype, :pyjlintegertype, :pyjlrationaltype, :pyjlrealtype, :pyjlcomplextype, :pyjlsettype, :pyjltypetype]
    @eval const $k = _jlwrap.$k
end

end
