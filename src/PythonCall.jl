module PythonCall

const VERSION = v"0.9.15"
const ROOT_DIR = dirname(@__DIR__)

include("Utils/Utils.jl")
include("C/C.jl")
include("GC/GC.jl")
include("Core/Core.jl")
include("Convert/Convert.jl")
include("PyMacro/PyMacro.jl")
include("Wrap/Wrap.jl")
include("JlWrap/JlWrap.jl")
include("Compat/Compat.jl")

# re-export everything
for m in [:Core, :Convert, :PyMacro, :Wrap, :JlWrap, :Compat]
    for k in names(@eval($m))
        if k != m
            @eval const $k = $m.$k
            @eval export $k
        end
    end
end

# non-exported API
for k in [:python_executable_path, :python_library_path, :python_library_handle, :python_version]
    @eval const $k = C.$k
end
for k in [:pynew, :pyisnull, :pycopy!, :getptr, :pydel!, :unsafe_pynext, :PyNULL, :CONFIG]
    @eval const $k = Core.$k
end
for k in [:pyconvert_add_rule, :pyconvert_return, :pyconvert_unconverted, :PYCONVERT_PRIORITY_WRAP, :PYCONVERT_PRIORITY_ARRAY, :PYCONVERT_PRIORITY_CANONICAL, :PYCONVERT_PRIORITY_NORMAL, :PYCONVERT_PRIORITY_FALLBACK]
    @eval const $k = Convert.$k
end
for k in [:event_loop_on, :event_loop_off, :fix_qt_plugin_path]
    @eval const $k = Compat.$k
end

# not API but used in tests
for k in [:pyjlanytype, :pyjlarraytype, :pyjlvectortype, :pyjlbinaryiotype, :pyjltextiotype, :pyjldicttype, :pyjlmoduletype, :pyjlintegertype, :pyjlrationaltype, :pyjlrealtype, :pyjlcomplextype, :pyjlsettype, :pyjltypetype]
    @eval const $k = JlWrap.$k
end

end
