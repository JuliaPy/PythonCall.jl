"""
    module CPython

This module provides a direct interface to the Python C API.
"""
module C

import Base: @kwdef
using Conda, Libdl, Requires

include("consts.jl")
include("pointers.jl")
include("context.jl")
include("gil.jl")

__init__() = init_context()

end
