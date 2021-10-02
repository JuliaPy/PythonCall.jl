struct UnsafePyObject
    ptr :: C.PyPtr
end

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
- `R`: The element type of the underlying buffer. Equal to `T` for scalar numeric types.
"""
struct PyArray{T,N,M,L,R} <: AbstractArray{T,N}
    ptr::Ptr{R}             # pointer to the data
    length::Int             # length of the array
    size::NTuple{N,Int}     # size of the array
    strides::NTuple{N,Int}  # strides (in bytes) between elements
    py::Py                  # underlying python object
    handle::Py              # the data in this array is valid as long as this handle is alive
    function PyArray{T,N,M,L,R}(::Val{:new}, ptr::Ptr{R}, size::NTuple{N,Int}, strides::NTuple{N,Int}, py::Py, handle::Py) where {T,N,M,L,R}
        T isa DataType || error("T must be a DataType")
        N isa Int || error("N must be an Int")
        M isa Bool || error("M must be a Bool")
        L isa Bool || error("L must be a Bool")
        R isa DataType || error("R must be a DataType")
        new{T,N,M,L,R}(ptr, prod(size), size, strides, py, handle)
    end
end
export PyArray

for N in (missing, 1, 2)
    for M in (missing, true, false)
        for L in (missing, true, false)
            name = Symbol(
                "Py",
                M === missing ? "" : M ? "Mutable" : "Immutable",
                L === missing ? "" : L ? "Linear" : "Cartesian",
                N === missing ? "Array" : N == 1 ? "Vector" : "Matrix",
            )
            name == :PyArray && continue
            vars = Any[:T, N===missing ? :N : N, M===missing ? :M : M, L===missing ? :L : L, :R]
            @eval const $name{$([v for v in vars if v isa Symbol]...)} = PyArray{$(vars...)}
            @eval export $name
        end
    end
end

(::Type{A})(x; array::Bool=true, buffer::Bool=true, copy::Bool=true) where {A<:PyArray} = @autopy x begin
    r = pyarray_make(A, x_, array=array, buffer=buffer, copy=copy)
    if pyconvert_isunconverted(r)
        error("cannot convert this Python '$(pytype(x_).__name__)' to a '$A'")
    else
        return pyconvert_result(r)::A
    end
end

pyconvert_rule_array_nocopy(::Type{A}, x::Py) where {A<:PyArray} = pyarray_make(A, x, copy=false)

function pyconvert_rule_array(::Type{A}, x::Py) where {A<:AbstractArray}
    r = pyarray_make(PyArray, x)
    if pyconvert_isunconverted(r)
        return pyconvert_unconverted()
    else
        return pyconvert_tryconvert(A, pyconvert_result(PyArray, r))
    end
end

abstract type PyArraySource end

function pyarray_make(::Type{A}, x::Py; array::Bool=true, buffer::Bool=true, copy::Bool=true) where {A<:PyArray}
    A == Union{} && return pyconvert_unconverted()
    if array && pyhasattr(x, "__array_struct__")
        try
            return pyarray_make(A, x, PyArraySource_ArrayStruct(x))
        catch exc
            @debug "failed to make PyArray from __array_struct__" exc=exc
        end
    end
    if array && pyhasattr(x, "__array_interface__")
        try
            return pyarray_make(A, x, PyArraySource_ArrayInterface(x))
        catch exc
            @debug "failed to make PyArray from __array_interface__" exc=exc
        end
    end
    if buffer && C.PyObject_CheckBuffer(getptr(x))
        try
            return pyarray_make(A, x, PyArraySource_Buffer(x))
        catch exc
            @debug "failed to make PyArray from buffer" exc=exc
        end
    end
    if copy && array && pyhasattr(x, "__array__")
        y = x.__array__()
        if pyhasattr(y, "__array_struct__")
            try
                return pyarray_make(A, y, PyArraySource_ArrayStruct(y))
            catch exc
                @debug "failed to make PyArray from __array__().__array_struct__" exc=exc
            end
        end
        if pyhasattr(y, "__array_interface__")
            try
                return pyarray_make(A, y, PyArraySource_ArrayInterface(y))
            catch exc
                @debug "failed to make PyArray from __array__().__array_interface__" exc=exc
            end
        end
    end
    return pyconvert_unconverted()
end

function pyarray_make(::Type{A}, x::Py, info::PyArraySource, ::Type{PyArray{T0,N0,M0,L0,R0}}=Utils._type_lb(A), ::Type{PyArray{T1,N1,M1,L1,R1}}=Utils._type_ub(A)) where {A<:PyArray,T0,N0,M0,L0,R0,T1,N1,M1,L1,R1}
    # R (buffer eltype)
    R′ = pyarray_get_R(info)::DataType
    if R0 == R1
        R = R1
        R == R′ || error("incorrect R, got $R, should be $R′")
    elseif T0 == T1 && T1 in (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64})
        R = T1
        R == R′ || error("incorrect R, got $R, should be $R′")
        R <: R1 || error("R out of bounds, got $R, should be <: $R1")
        R >: R0 || error("R out of bounds, got $R, should be >: $R0")
    else
        R = R′
    end
    # ptr
    ptr = pyarray_get_ptr(info, R)::Ptr{R}
    # T (eltype)
    if T0 == T1
        T = T1
        pyarray_check_T(T, R)
    else
        T = pyarray_get_T(R, T0, T1)::DataType
        T <: T1 || error("T out of bounds, got $T, should be <: $T1")
        T >: T0 || error("T out of bounds, got $T, should be >: $T0")
    end
    # N (ndims)
    N′ = pyarray_get_N(info)::Int
    if N0 == N1
        N = N1 isa Int ? N1 : Int(N1)
        N == N′ || error("incorrect N, got $N, should be $N′")
    else
        N = N′
    end
    # size
    size = pyarray_get_size(info, Val(N))::NTuple{N,Int}
    # strides
    strides = pyarray_get_strides(info, Val(N), R, size)::NTuple{N,Int}
    # M (mutable)
    # TODO: if M==false, we don't need to compute M′
    M′ = pyarray_get_M(info)::Bool
    if M0 == M1
        M = M1 isa Bool ? M1 : Bool(M1)
        M && !M′ && error("incorrect M, got $M, should be $M′")
    else
        M = M′
    end
    # L (linearly indexable)
    # TODO: if L==false, we don't need to compute L′
    L′ = N < 2 || strides == Utils.size_to_fstrides(strides[1], size)
    if L0 == L1
        L = L1 isa Bool ? L1 : Bool(L1)
        L && !L′ && error("incorrect L, got $L, should be $L′")
    else
        L = L′
    end
    # handle
    handle = pyarray_get_handle(info)
    # done
    arr = PyArray{T,N,M,L,R}(Val(:new), ptr, size, strides, x, handle)
    return pyconvert_return(arr)
end

# array interface

struct PyArraySource_ArrayInterface <: PyArraySource
    obj :: Py
    dict :: Py
    ptr :: Ptr{Cvoid}
    readonly :: Bool
    handle :: Py
end
function PyArraySource_ArrayInterface(x::Py)
    d = x.__array_interface__
    # offset
    # TODO: how is the offset measured?
    offset = pyconvert_and_del(Int, @py d.get("offset", 0))
    offset == 0 || error("not supported: non-zero offset")
    # mask
    @py("mask" in d) && error("not supported: mask")
    # data
    data = @py d.get("data")
    if pyistuple(data)
        ptr = Ptr{Cvoid}(pyconvert_and_del(UInt, data[0]))
        readonly = pyconvert_and_del(Bool, data[1])
        pydel!(data)
        handle = Py((x, d))
    else
        memview = @py memoryview(data === None ? x : data)
        pydel!(data)
        buf = UnsafePtr(C.PyMemoryView_GET_BUFFER(getptr(memview)))
        ptr = buf.buf[!]
        readonly = buf.readonly[] != 0
        handle = Py((x, memview))
    end
    PyArraySource_ArrayInterface(x, d, ptr, readonly, handle)
end

pyarray_typestrdescr_to_type(ts::String, descr::Py) = begin
    # byte swapped?
    bsc = ts[1]
    bs =
        bsc == '<' ? !Utils.islittleendian() :
        bsc == '>' ? Utils.islittleendian() :
        bsc == '|' ? false : error("endianness character not supported: $ts")
    bs && error("byte-swapping not supported: $ts")
    # element type
    etc = ts[2]
    if etc == 'b'
        sz = parse(Int, ts[3:end])
        sz == sizeof(Bool) && return Bool
        error("bool of this size not supported: $ts")
    elseif etc == 'i'
        sz = parse(Int, ts[3:end])
        sz == 1 && return Int8
        sz == 2 && return Int16
        sz == 4 && return Int32
        sz == 8 && return Int64
        sz == 16 && return Int128
        error("signed int of this size not supported: $ts")
    elseif etc == 'u'
        sz = parse(Int, ts[3:end])
        sz == 1 && return UInt8
        sz == 2 && return UInt16
        sz == 4 && return UInt32
        sz == 8 && return UInt64
        sz == 16 && return UInt128
        error("unsigned int of this size not supported: $ts")
    elseif etc == 'f'
        sz = parse(Int, ts[3:end])
        sz == 2 && return Float16
        sz == 4 && return Float32
        sz == 8 && return Float64
        error("float of this size not supported: $ts")
    elseif etc == 'c'
        sz = parse(Int, ts[3:end])
        sz == 4 && return Complex{Float16}
        sz == 8 && return Complex{Float32}
        sz == 16 && return Complex{Float64}
        error("complex of this size not supported: $ts")
    elseif etc == 'U'
        sz = parse(Int, ts[3:end])
        return Utils.StaticString{UInt32,sz}
    elseif etc == 'O'
        return UnsafePyObject
    else
        error("type not supported: $ts")
    end
end

function pyarray_get_R(src::PyArraySource_ArrayInterface)
    typestr = pyconvert_and_del(String, src.dict["typestr"])
    descr = @py @jl(src.dict).get("descr")
    R = pyarray_typestrdescr_to_type(typestr, descr)::DataType
    pydel!(descr)
    return R
end

pyarray_get_ptr(src::PyArraySource_ArrayInterface, ::Type{R}) where {R} = Ptr{R}(src.ptr)

pyarray_get_N(src::PyArraySource_ArrayInterface) = Int(@py jllen(@jl(src.dict)["shape"]))

pyarray_get_size(src::PyArraySource_ArrayInterface, ::Val{N}) where {N} = pyconvert_and_del(NTuple{N,Int}, src.dict["shape"])

function pyarray_get_strides(src::PyArraySource_ArrayInterface, ::Val{N}, ::Type{R}, size::NTuple{N,Int}) where {R,N}
    @py strides = @jl(src.dict).get("strides")
    if pyisnone(strides)
        pydel!(strides)
        return Utils.size_to_cstrides(sizeof(R), size)
    else
        return pyconvert_and_del(NTuple{N,Int}, strides)
    end
end

pyarray_get_M(src::PyArraySource_ArrayInterface) = !src.readonly

pyarray_get_handle(src::PyArraySource_ArrayInterface) = src.handle

# TODO: array struct

struct PyArraySource_ArrayStruct <: PyArraySource
    obj :: Py
    capsule :: Py
end
PyArraySource_ArrayStruct(x::Py) = PyArraySource_ArrayStruct(x, x.__array_struct__)

# TODO: buffer protocol

struct PyArraySource_Buffer <: PyArraySource
    obj :: Py
    memview :: Py
    buf :: C.UnsafePtr{C.Py_buffer}
end
function PyArraySource_Buffer(x::Py)
    memview = pybuiltins.memoryview(x)
    buf = C.UnsafePtr(C.PyMemoryView_GET_BUFFER(getptr(memview)))
    PyArraySource_Buffer(x, memview, buf)
end

pyarray_bufferformat_to_type(fmt::String) =
    fmt == "b" ? Cchar :
    fmt == "B" ? Cuchar :
    fmt == "h" ? Cshort :
    fmt == "H" ? Cushort :
    fmt == "i" ? Cint :
    fmt == "I" ? Cuint :
    fmt == "l" ? Clong :
    fmt == "L" ? Culong :
    fmt == "q" ? Clonglong :
    fmt == "Q" ? Culonglong :
    fmt == "e" ? Float16 :
    fmt == "f" ? Cfloat :
    fmt == "d" ? Cdouble :
    fmt == "?" ? Bool :
    fmt == "P" ? Ptr{Cvoid} :
    fmt == "O" ? UnsafePyObject :
    fmt == "=e" ? Float16 :
    fmt == "=f" ? Float32 :
    fmt == "=d" ? Float64 :
    error("not implemented: $(repr(fmt))")

function pyarray_get_R(src::PyArraySource_Buffer)
    ptr = src.buf.format[]
    return ptr == C_NULL ? UInt8 : pyarray_bufferformat_to_type(String(ptr))
end

pyarray_get_ptr(src::PyArraySource_Buffer, ::Type{R}) where {R} = Ptr{R}(src.buf.buf[!])

pyarray_get_N(src::PyArraySource_Buffer) = Int(src.buf.ndim[])

function pyarray_get_size(src::PyArraySource_Buffer, ::Val{N}) where {N}
    size = src.buf.shape[]
    if size == C_NULL
        N == 0 ? () : N == 1 ? (Int(src.buf.len[]),) : @assert false
    else
        ntuple(i->Int(size[i]), N)
    end
end

function pyarray_get_strides(src::PyArraySource_Buffer, ::Val{N}, ::Type{R}, size::NTuple{N,Int}) where {N,R}
    strides = src.buf.strides[]
    if strides == C_NULL
        itemsize = src.buf.shape[] == C_NULL ? 1 : src.buf.itemsize[]
        Utils.size_to_cstrides(itemsize, size)
    else
        ntuple(i->Int(strides[i]), N)
    end
end

pyarray_get_M(src::PyArraySource_Buffer) = src.buf.readonly[] == 0

pyarray_get_handle(src::PyArraySource_Buffer) = src.memview

# AbstractArray methods

Base.length(x::PyArray) = x.length

Base.size(x::PyArray) = x.size

Utils.ismutablearray(x::PyArray{T,N,M,L,R}) where {T,N,M,L,R} = M

Base.IndexStyle(::Type{PyArray{T,N,M,L,R}}) where {T,N,M,L,R} = L ? Base.IndexLinear() : Base.IndexCartesian()

@propagate_inbounds Base.getindex(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} = pyarray_getindex(x, i...)
@propagate_inbounds Base.getindex(x::PyArray{T,N,M,true}, i::Int) where {T,N,M} = pyarray_getindex(x, i)
@propagate_inbounds Base.getindex(x::PyArray{T,1,M,true}, i::Int) where {T,M} = pyarray_getindex(x, i)

@propagate_inbounds function pyarray_getindex(x::PyArray, i...)
    @boundscheck checkbounds(x, i...)
    pyarray_load(eltype(x), x.ptr + pyarray_offset(x, i...))
end

@propagate_inbounds Base.setindex!(x::PyArray{T,N,true}, v, i::Vararg{Int,N}) where {T,N} = pyarray_setindex!(x, v, i...)
@propagate_inbounds Base.setindex!(x::PyArray{T,N,true,true}, v, i::Int) where {T,N} = pyarray_setindex!(x, v, i)
@propagate_inbounds Base.setindex!(x::PyArray{T,1,true,true}, v, i::Int) where {T} = pyarray_setindex!(x, v, i)

@propagate_inbounds function pyarray_setindex!(x::PyArray{T,N,true}, v, i...) where {T,N}
    @boundscheck checkbounds(x, i...)
    pyarray_store!(x.ptr + pyarray_offset(x, i...), convert(T, v))
    x
end

pyarray_offset(x::PyArray{T,N,M,true}, i::Int) where {T,N,M} = N == 0 ? 0 : (i - 1) * x.strides[1]
pyarray_offset(x::PyArray{T,1,M,true}, i::Int) where {T,M} = (i - 1) .* x.strides[1]
pyarray_offset(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} = sum((i .- 1) .* x.strides)
pyarray_offset(x::PyArray{T,0}) where {T} = 0

pyarray_load(::Type{R}, p::Ptr{R}) where {R} = unsafe_load(p)
pyarray_load(::Type{Py}, p::Ptr{UnsafePyObject}) = begin
    o = unsafe_load(p)
    o.ptr == C_NULL ? Py(nothing) : pynew(incref(o.ptr))
end

pyarray_store!(p::Ptr{R}, x::R) where {R} = unsafe_store!(p, x)
pyarray_store!(p::Ptr{UnsafePyObject}, x::Py) = begin
    decref(unsafe_load(p).ptr)
    unsafe_store!(p, UnsafePyObject(GC.@preserve x incref(getptr(x))))
end

pyarray_get_T(::Type{R}, ::Type{T0}, ::Type{T1}) where {R,T0,T1} = T0 <: R <: T1 ? R : error("not possible")
pyarray_get_T(::Type{UnsafePyObject}, ::Type{T0}, ::Type{T1}) where {T0,T1} = T0 <: Py <: T1 ? Py : T0 <: UnsafePyObject <: T1 ? UnsafePyObject : error("not possible")

pyarray_check_T(::Type{T}, ::Type{R}) where {T,R} = T == R ? nothing : error("invalid eltype T=$T for raw eltype R=$R")
pyarray_check_T(::Type{Py}, ::Type{UnsafePyObject}) = nothing
