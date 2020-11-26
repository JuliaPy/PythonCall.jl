module Python

using Dates, UnsafePointers, Libdl, Conda, Tables, TableTraits, IteratorInterfaceExtensions, Markdown, Base64
using Base: @kwdef

# things not directly dependent on PyObject or libpython
include("utils.jl")

# Global configuration
# CONFIG gets populated by __init__
@kwdef mutable struct Config
    dlopenflags :: UInt32 = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL
    exepath :: Union{String,Nothing} = nothing
    libpath :: Union{String,Nothing} = nothing
    libptr :: Ptr{Cvoid} = C_NULL
    pyhome :: Union{String,Nothing} = nothing
    pyhome_w :: Vector{Cwchar_t} = []
    pyprogname :: Union{String,Nothing} = nothing
    pyprogname_w :: Vector{Cwchar_t} = []
    isstackless :: Bool = false
    preloaded :: Bool = false
    preinitialized :: Bool = false
    isinitialized :: Bool = false
    version :: VersionNumber = VersionNumber(0)
    isconda :: Bool = false
end
Base.show(io::IO, ::MIME"text/plain", c::Config) =
    for k in fieldnames(Config)
        println(io, k, ": ", repr(getfield(c, k)))
    end
const CONFIG = Config()

# C API
include("cpython.jl")

const C = CPython
const CPyPtr = C.PyPtr
struct CPyObjRef
    ptr :: CPyPtr
end

# initialize
include("init.jl")

# core
include("object.jl")
include("error.jl")
include("import.jl")

# abstract interfaces
include("number.jl")
include("sequence.jl")

# fundamental objects
include("type.jl")
include("none.jl")

# numeric objects
include("bool.jl")
include("int.jl")
include("float.jl")
include("complex.jl")

# sequence objects
include("str.jl")
include("bytes.jl")
include("bytearray.jl")
include("tuple.jl")
include("list.jl")

# mapping objects
include("dict.jl")
include("set.jl")

# function objects
include("function.jl")

# other objects
include("slice.jl")
include("range.jl")

# standard library
include("builtins.jl")
include("eval.jl")
include("stdlib.jl")
include("io.jl")
include("fraction.jl")
include("datetime.jl")
include("collections.jl")

# other packages
include("pandas.jl")
include("numpy.jl")
include("matplotlib.jl")

# other Julia wrappers around Python values
include("PyIterable.jl")
include("PyList.jl")
include("PyDict.jl")
include("PyObjectArray.jl")
include("PyBuffer.jl")
include("PyArray.jl")
include("PyIO.jl")

# other functionality
include("convert.jl")
include("newtype.jl")
include("julia.jl")
include("base.jl")
include("pywith.jl")
include("gui.jl")

end # module
