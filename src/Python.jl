module Python

using Dates, UnsafePointers, Libdl, Conda, Tables, TableTraits, IteratorInterfaceExtensions, Markdown, Base64, Requires
using Base: @kwdef

# things not directly dependent on PyObject or libpython
include("utils.jl")

# Global configuration
# CONFIG gets populated by __init__
@kwdef mutable struct Config
    "Flags used to dlopen the Python library."
    dlopenflags :: UInt32 = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL
    "Path to the Python executable."
    exepath :: Union{String,Nothing} = nothing
    "Path to the Python library."
    libpath :: Union{String,Nothing} = nothing
    "Handle to the open Python library."
    libptr :: Ptr{Cvoid} = C_NULL
    "Used to set the Python prefix."
    pyhome :: Union{String,Nothing} = nothing
    "pyhome as a Cwstring"
    pyhome_w :: Vector{Cwchar_t} = []
    "Used to set the Python program name."
    pyprogname :: Union{String,Nothing} = nothing
    "pyprogname as a Cwstring"
    pyprogname_w :: Vector{Cwchar_t} = []
    "True if this is stackless Python."
    isstackless :: Bool = false
    """True if Julia is embedded into Python (indicated by ENV["PYTHONJL_LIBPTR"] being set)."""
    isembedded :: Bool = false
    "True if the Python interpreter is currently initialized."
    isinitialized :: Bool = false
    "The running Python version."
    version :: VersionNumber = VersionNumber(0)
    "True if this is the Python in some Conda environment."
    isconda :: Bool = false
    "If `isconda` is true, this is the Conda environment path."
    condaenv :: String = Conda.ROOTENV
    "When true, automatically calls `pyplotshow` each time a notebook cell is evaluated."
    pyplotautoshow :: Bool = true
    "When true, automatically calls `fix_qt_plugin_path` when `PyQt5`, `PyQt4`, `PySide` or `PySide2` is loaded."
    qtfix :: Bool = true
    "When true, automatically sets `sys.last_traceback` when a Python exception is printed, so that `pdb.pm()` works."
    sysautolasttraceback :: Bool = true
    "True if the Python input hook is currently running."
    inputhookrunning :: Bool = false
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

# core
include("object.jl")
include("error.jl")
include("import.jl")
include("gil.jl")

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
include("PySet.jl")
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

# initialize
include("init.jl")

end # module
