# Convert

@enum PyConvertPriority begin
    PYCONVERT_PRIORITY_WRAP = 400
    PYCONVERT_PRIORITY_ARRAY = 300
    PYCONVERT_PRIORITY_CANONICAL = 200
    PYCONVERT_PRIORITY_NORMAL = 0
    PYCONVERT_PRIORITY_FALLBACK = -100
end

# Core

"""
    Py(x)

Convert `x` to a Python object, of type `Py`.

Conversion happens according to [these rules](@ref jl2py-conversion).

Such an object supports attribute access (`obj.attr`), indexing (`obj[idx]`), calling
(`obj(arg1, arg2)`), iteration (`for x in obj`), arithmetic (`obj + obj2`) and comparison
(`obj > obj2`), among other things. These operations convert all their arguments to `Py` and
return `Py`.
"""
mutable struct Py
    ptr::Ptr{Cvoid}
    Py(::Val{:new}, ptr::Ptr) = finalizer(Core.py_finalizer, new(Ptr{Cvoid}(ptr)))
end

"""
    PyException(x)

Wraps the Python exception `x` as a Julia `Exception`.
"""
mutable struct PyException <: Exception
    _t::Py
    _v::Py
    _b::Py
    _isnormalized::Bool
end

"""
    pybuiltins

An object whose fields are the Python builtins, of type [`Py`](@ref PythonCall.Py).

For example `pybuiltins.None`, `pybuiltins.int`, `pybuiltins.ValueError`.
"""
baremodule pybuiltins end

# Wrap

"""
    PyArray{T,N,M,L,R}(x; copy=true, array=true, buffer=true)

Wrap the Python array `x` as a Julia `AbstractArray{T,N}`.

The input `x` can be `bytes`, `bytearray`, `array.array`, `numpy.ndarray` or anything satisfying the buffer protocol (if `buffer=true`) or the numpy array interface (if `array=true`).

If `copy=false` then the resulting array is guaranteed to directly wrap the data in `x`. If `copy=true` then a copy is taken if necessary to produce an array.

The type parameters are all optional, and are:
- `T`: The element type.
- `N`: The number of dimensions.
- `M`: True if the array is mutable.
- `L`: True if the array supports fast linear indexing.
- `R`: The element type of the underlying buffer. Often equal to `T`.
"""
struct PyArray{T,N,M,L,R} <: AbstractArray{T,N}
    ptr::Ptr{R}             # pointer to the data
    length::Int             # length of the array
    size::NTuple{N,Int}     # size of the array
    strides::NTuple{N,Int}  # strides (in bytes) between elements
    py::Py                  # underlying python object
    handle::Any             # the data in this array is valid as long as this handle is alive
    function PyArray{T,N,M,L,R}(
        ::Val{:new},
        ptr::Ptr{R},
        size::NTuple{N,Int},
        strides::NTuple{N,Int},
        py::Py,
        handle::Any,
    ) where {T,N,M,L,R}
        T isa Type || error("T must be a Type")
        N isa Int || error("N must be an Int")
        M isa Bool || error("M must be a Bool")
        L isa Bool || error("L must be a Bool")
        R isa DataType || error("R must be a DataType")
        new{T,N,M,L,R}(ptr, prod(size), size, strides, py, handle)
    end
end

"""
    PyDict{K=Py,V=Py}([x])

Wraps the Python dict `x` (or anything satisfying the mapping interface) as an `AbstractDict{K,V}`.

If `x` is not a Python object, it is converted to one using `pydict`.
"""
struct PyDict{K,V} <: AbstractDict{K,V}
    py::Py
    PyDict{K,V}(x = pydict()) where {K,V} = new{K,V}(ispy(x) ? Py(x) : pydict(x))
end

"""
    PyIO(x; own=false, text=missing, line_buffering=false, buflen=4096)

Wrap the Python IO stream `x` as a Julia IO stream.

When this goes out of scope and is finalized, it is automatically flushed. If `own=true` then it is also closed.

If `text=false` then `x` must be a binary stream and arbitrary binary I/O is possible.
If `text=true` then `x` must be a text stream and only UTF-8 must be written (i.e. use `print` not `write`).
If `text` is not specified then it is chosen automatically.
If `x` is a text stream and you really need a binary stream, then often `PyIO(x.buffer)` will work.

If `line_buffering=true` then output is flushed at each line.

For efficiency, reads and writes are buffered before being sent to `x`.
The size of the buffers is `buflen`.
The buffers are cleared using `flush`.
"""
mutable struct PyIO <: IO
    py::Py
    # true to close the file automatically
    own::Bool
    # true if `o` is text, false if binary
    text::Bool
    # true to flush whenever '\n' or '\r' is encountered
    line_buffering::Bool
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof::Bool
    # input buffer
    ibuflen::Int
    ibuf::Vector{UInt8}
    # output buffer
    obuflen::Int
    obuf::Vector{UInt8}

    function PyIO(
        x;
        own::Bool = false,
        text::Union{Missing,Bool} = missing,
        buflen::Integer = 4096,
        ibuflen::Integer = buflen,
        obuflen::Integer = buflen,
        line_buffering::Bool = false,
    )
        if text === missing
            text = pyhasattr(x, "encoding")
        end
        buflen = convert(Int, buflen)
        buflen > 0 || error("buflen must be positive")
        ibuflen = convert(Int, ibuflen)
        ibuflen > 0 || error("ibuflen must be positive")
        obuflen = convert(Int, obuflen)
        obuflen > 0 || error("obuflen must be positive")
        new(Py(x), own, text, line_buffering, false, ibuflen, UInt8[], obuflen, UInt8[])
    end
end

"""
    PyIterable{T=Py}(x)

This object iterates over iterable Python object `x`, yielding values of type `T`.
"""
struct PyIterable{T}
    py::Py
    PyIterable{T}(x) where {T} = new{T}(Py(x))
end

"""
    PyList{T=Py}([x])

Wraps the Python list `x` (or anything satisfying the sequence interface) as an `AbstractVector{T}`.

If `x` is not a Python object, it is converted to one using `pylist`.
"""
struct PyList{T} <: AbstractVector{T}
    py::Py
    PyList{T}(x = pylist()) where {T} = new{T}(ispy(x) ? Py(x) : pylist(x))
end

"""
    PyTable(x)

Wrap `x` as a Tables.jl-compatible table.

`PyTable` is an abstract type. See [`PyPandasDataFrame`](@ref) for a concrete example.
"""
abstract type PyTable end

"""
    PyPandasDataFrame(x; [indexname::Union{Nothing,Symbol}], [columnnames::Function], [columntypes::Function])

Wraps the pandas DataFrame `x` as a Tables.jl-compatible table.

- `indexname`: The name of the column including the index. The default is `nothing`, meaning
  to exclude the index.
- `columnnames`: A function mapping the Python column name (a `Py`) to the Julia one (a
  `Symbol`). The default is `x -> Symbol(x)`.
- `columntypes`: A function taking the column name (a `Symbol`) and returning either the
  desired element type of the column, or `nothing` to indicate automatic inference.
"""
struct PyPandasDataFrame <: PyTable
    py::Py
    indexname::Union{Symbol,Nothing}
    columnnames::Function # Py -> Symbol
    columntypes::Function # Symbol -> Union{Type,Nothing}
    function PyPandasDataFrame(
        x;
        indexname::Union{Symbol,Nothing} = nothing,
        columnnames::Function = x -> Symbol(x),
        columntypes::Function = x -> nothing,
    )
        new(Py(x), indexname, columnnames, columntypes)
    end
end

"""
    PySet{T=Py}([x])

Wraps the Python set `x` (or anything satisfying the set interface) as an `AbstractSet{T}`.

If `x` is not a Python object, it is converted to one using `pyset`.
"""
struct PySet{T} <: AbstractSet{T}
    py::Py
    PySet{T}(x = pyset()) where {T} = new{T}(ispy(x) ? Py(x) : pyset(x))
end

"""
    PyObjectArray(undef, dims...)
    PyObjectArray(array)

An array of `Py`s which supports the Python buffer protocol.

Internally, the objects are stored as an array of pointers.
"""
mutable struct PyObjectArray{N} <: AbstractArray{Py,N}
    ptrs::Array{Ptr{Cvoid},N}
    function PyObjectArray{N}(::UndefInitializer, dims::NTuple{N,Integer}) where {N}
        x = new{N}(fill(C_NULL, dims))
        finalizer(JlWrap.pyobjectarray_finalizer, x)
    end
end
const PyObjectVector = PyObjectArray{1}
const PyObjectMatrix = PyObjectArray{2}
