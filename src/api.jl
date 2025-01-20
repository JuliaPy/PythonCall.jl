"The version of PythonCall."
const VERSION = v"0.9.23"

# public types
include("types-api.jl")

# public submodules
include("GIL-api.jl")
include("GC-api.jl")

# config
@kwdef mutable struct Config
    meta::String = ""
    auto_sys_last_traceback::Bool = true
    auto_fix_qt_plugin_path::Bool = true
end

const CONFIG = Config()

# exported functions
for name in [:ispy]
    @eval function $name end
    @eval export $name
end

# public functions
for name in [
    :python_executable_path,
    :python_library_handle,
    :python_library_path,
    :python_version,
    :pynew,
    :pyisnull,
    :pycopy!,
    :getptr,
    :pydel!,
    :unsafe_pynext,
    :pyconvert_add_rule,
    :pyconvert_return,
    :pyconvert_unconverted,
    :event_loop_on,
    :event_loop_off,
    :fix_qt_plugin_path,
]
    @eval function $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end

# other exported bindings
for name in [:Py, :PyException]
    @eval export $name
end

# other public bindings
for name in [
    :VERSION,
    :GIL,
    :GC,
    :CONFIG,
    # :PyNULL,
    # :PYCONVERT_PRIORITY_WRAP,
    # :PYCONVERT_PRIORITY_ARRAY,
    # :PYCONVERT_PRIORITY_CANONICAL,
    # :PYCONVERT_PRIORITY_NORMAL,
    # :PYCONVERT_PRIORITY_FALLBACK,
]
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end
