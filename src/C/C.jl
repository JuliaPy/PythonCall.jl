"""
    module PythonCall.C

This module provides a direct interface to the Python C API.
"""
module C

using ..Utils

using Base: @kwdef
using UnsafePointers: UnsafePtr
using Libdl:
    dlpath, dlopen, dlopen_e, dlclose, dlsym, dlsym_e, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL
using Preferences: @load_preference

# do not load CondaPkg if the exe preference is set to something else
if @load_preference("exe", "") in ("", "@CondaPkg")
    using CondaPkg: CondaPkg
end

import ..PythonCall:
    python_executable_path, python_library_path, python_library_handle, python_version

include("consts.jl")
include("pointers.jl")
include("extras.jl")
include("context.jl")
include("api.jl")

function __init__()
    init_context()
end

end
