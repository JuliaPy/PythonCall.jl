# Global configuration
# CONFIG gets populated by __init__
@kwdef mutable struct Config
    "Flags used to dlopen the Python library."
    dlopenflags::UInt32 = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL
    "Path to the Python executable."
    exepath::Union{String,Nothing} = nothing
    "Path to the Python library."
    libpath::Union{String,Nothing} = nothing
    "Handle to the open Python library."
    libptr::Ptr{Cvoid} = C_NULL
    "Used to set the Python prefix."
    pyhome::Union{String,Nothing} = nothing
    "pyhome as a Cwstring"
    pyhome_w::Vector{Cwchar_t} = []
    "Used to set the Python program name."
    pyprogname::Union{String,Nothing} = nothing
    "pyprogname as a Cwstring"
    pyprogname_w::Vector{Cwchar_t} = []
    "True if this is stackless Python."
    isstackless::Bool = false
    """True if Julia is embedded into Python (indicated by ENV["JULIA_PYTHONCALL_LIBPTR"] being set)."""
    isembedded::Bool = false
    "True if the Python interpreter is currently initialized."
    isinitialized::Bool = false
    "True if the Python interpreter was already initialized."
    preinitialized::Bool = false
    "The running Python version."
    version::VersionNumber = VersionNumber(0)
    "True if this is the Python in some Conda environment."
    isconda::Bool = false
    "If `isconda` is true, this is the Conda environment path."
    condaenv::String = ""
    "When true, automatically calls `pyplotshow` each time a notebook cell is evaluated."
    pyplotautoshow::Bool = true
    "When true, automatically calls `fix_qt_plugin_path` when `PyQt5`, `PyQt4`, `PySide` or `PySide2` is loaded."
    qtfix::Bool = true
    "When true, automatically sets `sys.last_traceback` when a Python exception is printed, so that `pdb.pm()` works."
    sysautolasttraceback::Bool = true
    "True if the Python input hook is currently running."
    inputhookrunning::Bool = false
    "When true, if being run inside an IPython kernel then integrate IO."
    ipythonintegration::Bool = true
    "When true, this is being run inside an IPython kernal."
    isipython::Bool = false
end
Base.show(io::IO, ::MIME"text/plain", c::Config) =
    for k in fieldnames(Config)
        println(io, k, ": ", repr(getfield(c, k)))
    end
const CONFIG = Config()

"""
    ispyreftype(::Type{T})

True if `T` is a wrapper type for a single Python reference.

Such objects must implement:
- `pyptr(o::T) =` the underlying pointer (or NULL on error)
- `unsafe_convert(::Type{CPyPtr}, o::T) = checknull(pyptr(o))`
"""
ispyreftype(::Type) = false

"""
    ispyref(x)

Equivalent to `ispyreftype(typeof(x))`.
"""
ispyref(x) = ispyreftype(typeof(x))

"""
    pyptr(o)

Retrieve the underlying Python object pointer from o.
"""
function pyptr end

convertref(::Type{T}, x) where {T} = ispyreftype(T) ? x : convert(T, x)
tryconvertref(::Type{T}, x) where {T} = ispyreftype(T) ? x : tryconvert(T, x)

function pythrow end

function pyref_finalize! end

"""
    PyLazyObject(x)

Convert `x` to a Python object lazily, i.e. the conversion happens the first time the object is accessed.
"""
mutable struct PyLazyObject{T}
    ptr::Ptr{Cvoid}
    val::T
    PyLazyObject(val::T) where {T} = finalizer(pyref_finalize!, new{T}(C_NULL, val))
end

"""
    PyInternedString(x::String)

Convert `x` to an interned Python string.

This can provide a performance boost when using strings to index dictionaries or get attributes.

See also [`@pystr_str`](@ref).
"""
mutable struct PyInternedString
    ptr::Ptr{Cvoid}
    val::String
    PyInternedString(x::String) = finalizer(pyref_finalize!, new(C_NULL, x))
end
