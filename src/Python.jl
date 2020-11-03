module Python

using Dates, UnsafePointers, Libdl, Conda
using Base: @kwdef

# dependencies
include(joinpath(@__DIR__, "..", "deps", "deps.jl"))

# C API
include("ctypedefs.jl")
include("cconsts.jl")
include("cbuffer.jl")
include("cattrstructs.jl")
include("cmethodstructs.jl")
include("cpyobject.jl")
include("cpycall.jl")
include("init.jl")

# core
include("object.jl")
include("cpycall2.jl")
include("error.jl")
include("import.jl")
include("builtins.jl")
include("convert.jl")

# abstract objects
include("number.jl")

# concrete objects
include("type.jl")
include("none.jl")

include("bool.jl")
include("int.jl")
include("float.jl")
include("complex.jl")
include("fraction.jl")

include("str.jl")
include("bytes.jl")
include("bytearray.jl")
include("tuple.jl")
include("list.jl")
include("dict.jl")
include("set.jl")

include("slice.jl")
include("range.jl")
include("datetime.jl")

# extra api
include("newtype.jl")
include("julia.jl")
include("base.jl")
include("pywith.jl")
include("PyList.jl")
include("PyDict.jl")
include("PyObjectArray.jl")

end # module
