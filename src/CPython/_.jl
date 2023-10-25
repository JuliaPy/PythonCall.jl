"""
    module _CPython

This module provides a direct interface to the Python C API.
"""
module _CPython

using Base: @kwdef
using UnsafePointers: UnsafePtr
using CondaPkg: CondaPkg
using Pkg: Pkg
using Requires: @require
using Libdl: dlpath, dlopen, dlopen_e, dlclose, dlsym, dlsym_e, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL

# import Base: @kwdef
# import CondaPkg
# import Pkg
# using Libdl, Requires, UnsafePointers, Serialization, ..Utils

include("consts.jl")
include("pointers.jl")
include("extras.jl")
include("context.jl")
include("gil.jl")

function __init__()
    init_context()
end

end
