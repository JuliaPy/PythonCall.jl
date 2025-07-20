"""
    module PythonCall.C

This module provides a direct interface to the Python C API.
"""
module C

using Base: @kwdef
using UnsafePointers: UnsafePtr
using CondaPkg: CondaPkg
using Pkg: Pkg
using Requires: @require
using Libdl:
    dlpath, dlopen, dlopen_e, dlclose, dlsym, dlsym_e, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL

import ..PythonCall: python_executable_path, python_library_path, python_library_handle, python_version


include("consts.jl")
include("pointers.jl")
include("extras.jl")
include("context.jl")
include("api.jl")

function __init__()
    init_context()
end

end
