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
- `R`: The element type of the underlying buffer. Often equal to `T`.
"""
struct PyArray{T,N,M,L,R} <: AbstractArray{T,N}
    ptr::Ptr{R}             # pointer to the data
    length::Int             # length of the array
    size::NTuple{N,Int}     # size of the array
    strides::NTuple{N,Int}  # strides (in bytes) between elements
    py::Py                  # underlying python object
    handle::Py              # the data in this array is valid as long as this handle is alive
    function PyArray{T,N,M,L,R}(::Val{:new}, ptr::Ptr{R}, size::NTuple{N,Int}, strides::NTuple{N,Int}, py::Py, handle::Py) where {T,N,M,L,R}
        T isa Type || error("T must be a Type")
        N isa Int || error("N must be an Int")
        M isa Bool || error("M must be a Bool")
        L isa Bool || error("L must be a Bool")
        R isa DataType || error("R must be a DataType")
        new{T,N,M,L,R}(ptr, prod(size), size, strides, py, handle)
    end
end
export PyArray

ispy(::PyArray) = true
Py(x::PyArray) = x.py

for N in (missing, 1, 2)
    for M in (missing, true, false)
        for L in (missing, true, false)
            for R in (true, false)
                name = Symbol(
                    "Py",
                    M === missing ? "" : M ? "Mutable" : "Immutable",
                    L === missing ? "" : L ? "Linear" : "Cartesian",
                        R ? "Raw" : "",
                    N === missing ? "Array" : N == 1 ? "Vector" : "Matrix",
                )
                name == :PyArray && continue
                    vars = Any[:T, N===missing ? :N : N, M===missing ? :M : M, L===missing ? :L : L, R ? :T : :R]
                    @eval const $name{$(unique([v for v in vars if v isa Symbol])...)} = PyArray{$(vars...)}
                @eval export $name
            end
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
    # TODO: try/catch is SLOW if an error is thrown, think about sending errors via return values instead
    A == Union{} && return pyconvert_unconverted()
    if array && (xa = pygetattr(x, "__array_struct__", PyNULL); !pyisnull(xa))
        try
            return pyarray_make(A, x, PyArraySource_ArrayStruct(x, xa))
        catch exc
            @debug "failed to make PyArray from __array_struct__" exc=exc
        end
    end
    if array && (xi = pygetattr(x, "__array_interface__", PyNULL); !pyisnull(xi))
        try
            return pyarray_make(A, x, PyArraySource_ArrayInterface(x, xi))
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
        if (ya = pygetattr(y, "__array_struct__", PyNULL); !pyisnull(ya))
            try
                return pyarray_make(A, y, PyArraySource_ArrayStruct(y, ya))
            catch exc
                @debug "failed to make PyArray from __array__().__array_interface__" exc=exc
            end
        end
        if (yi = pygetattr(y, "__array_interface__", PyNULL); !pyisnull(yi))
            try
                return pyarray_make(A, y, PyArraySource_ArrayInterface(y, yi))
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
    # elseif T0 == T1 && T1 in (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
    #     R = T1
    #     R == R′ || error("incorrect R, got $R, should be $R′")
    #     R <: R1 || error("R out of bounds, got $R, should be <: $R1")
    #     R >: R0 || error("R out of bounds, got $R, should be >: $R0")
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
function PyArraySource_ArrayInterface(x::Py, d::Py=x.__array_interface__)
    # offset
    # TODO: how is the offset measured?
    offset = pyconvert(Int, @py d.get("offset", 0))
    offset == 0 || error("not supported: non-zero offset")
    # mask
    @py("mask" in d) && error("not supported: mask")
    # data
    data = @py d.get("data")
    if pyistuple(data)
        ptr = Ptr{Cvoid}(pyconvert(UInt, data[0]))
        readonly = pyconvert(Bool, data[1])
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
    elseif etc == 'V'
        pyisnull(descr) && error("not supported: void dtype with null descr")
        sz = parse(Int, ts[3:end])
        T = pyarray_descr_to_type(descr)
        sizeof(T) == sz || error("size mismatch: itemsize=$sz but sizeof(descr)=$(sizeof(T))")
        return T
    else
        error("not supported: dtype of kind: $(repr(etc))")
    end
end

function pyarray_descr_to_type(descr::Py)
    fnames = Symbol[]
    foffsets = Int[]
    ftypes = DataType[]
    curoffset = 0
    for item in descr
        # get the name
        name = item[0]
        if pyistuple(name)
            name = name[0]
        end
        fname = Symbol(pyconvert(String, name))
        # get the shape
        if length(item) > 2
            shape = pyconvert(Vector{Int}, item[2])
        else
            shape = Int[]
        end
        # get the type
        descr2 = item[1]
        if pyisstr(descr2)
            typestr = pyconvert(String, descr2)
            # void entries are just padding to ignore
            if typestr[2] == 'V'
                curoffset += parse(Int, typestr[3:end]) * prod(shape)
                continue
            end
            ftype = pyarray_typestrdescr_to_type(typestr, PyNULL)
        else
            ftype = pyarray_descr_to_type(descr2)
        end
        # apply the shape
        for n in reverse(shape)
            ftype = NTuple{n,ftype}
        end
        # save the field
        push!(fnames, fname)
        push!(foffsets, curoffset)
        push!(ftypes, ftype)
        curoffset += sizeof(ftype)
    end
    # construct the tuple type and check its offsets and size
    # TODO: support non-aligned dtypes by packing them into a custom type and reinterpreting
    T = Tuple{ftypes...}
    for (i, o) in pairs(foffsets)
        fieldoffset(T, i) == o || error("not supported: dtype that is not aligned: $descr")
    end
    sizeof(T) == curoffset || error("not supported: dtype with end padding: $descr")
    # return the tuple type if the field names are f0, f1, ..., else return a named tuple
    if fnames == [Symbol(:f, i-1) for i in 1:length(fnames)]
        return T
    else
        return NamedTuple{Tuple(fnames), T}
    end
end

function pyarray_get_R(src::PyArraySource_ArrayInterface)
    typestr = pyconvert(String, src.dict["typestr"])
    descr = @py @jl(src.dict).get("descr")
    R = pyarray_typestrdescr_to_type(typestr, descr)::DataType
    pydel!(descr)
    return R
end

pyarray_get_ptr(src::PyArraySource_ArrayInterface, ::Type{R}) where {R} = Ptr{R}(src.ptr)

pyarray_get_N(src::PyArraySource_ArrayInterface) = Int(@py jllen(@jl(src.dict)["shape"]))

pyarray_get_size(src::PyArraySource_ArrayInterface, ::Val{N}) where {N} = pyconvert(NTuple{N,Int}, src.dict["shape"])

function pyarray_get_strides(src::PyArraySource_ArrayInterface, ::Val{N}, ::Type{R}, size::NTuple{N,Int}) where {R,N}
    @py strides = @jl(src.dict).get("strides")
    if pyisnone(strides)
        pydel!(strides)
        return Utils.size_to_cstrides(sizeof(R), size)
    else
        return pyconvert(NTuple{N,Int}, strides)
    end
end

pyarray_get_M(src::PyArraySource_ArrayInterface) = !src.readonly

pyarray_get_handle(src::PyArraySource_ArrayInterface) = src.handle

# array struct

struct PyArraySource_ArrayStruct <: PyArraySource
    obj :: Py
    capsule :: Py
    info :: C.PyArrayInterface
end
function PyArraySource_ArrayStruct(x::Py, capsule::Py=x.__array_struct__)
    name = C.PyCapsule_GetName(getptr(capsule))
    ptr = C.PyCapsule_GetPointer(getptr(capsule), name)
    info = unsafe_load(Ptr{C.PyArrayInterface}(ptr))
    @assert info.two == 2
    return PyArraySource_ArrayStruct(x, capsule, info)
end

function pyarray_get_R(src::PyArraySource_ArrayStruct)
    swapped = !Utils.isflagset(src.info.flags, C.NPY_ARRAY_NOTSWAPPED)
    hasdescr = Utils.isflagset(src.info.flags, C.NPY_ARR_HAS_DESCR)
    swapped && error("byte-swapping not supported")
    kind = src.info.typekind
    size = src.info.itemsize
    if kind == 98  # b = bool
        if size == sizeof(Bool)
            return Bool
        else
            error("bool of this size not supported: $size")
        end
    elseif kind == 105  # i = int
        if size == 1
            return Int8
        elseif size == 2
            return Int16
        elseif size == 4
            return Int32
        elseif size == 8
            return Int64
        else
            error("int of this size not supported: $size")
        end
    elseif kind == 117  # u = uint
        if size == 1
            return UInt8
        elseif size == 2
            return UInt16
        elseif size == 4
            return UInt32
        elseif size == 8
            return UInt64
        else
            error("uint of this size not supported: $size")
        end
    elseif kind == 102  # f = float
        if size == 2
            return Float16
        elseif size == 4
            return Float32
        elseif size == 8
            return Float64
        else
            error("float of this size not supported: $size")
        end
    elseif kind == 99  # c = complex
        if size == 4
            return ComplexF16
        elseif size == 8
            return ComplexF32
        elseif size == 16
            return ComplexF64
        end
    elseif kind == 109  # m = timedelta
        error("timedelta not supported")
    elseif kind == 77  # M = datetime
        return DateTime
    elseif kind == 79  # O = object
        if size == sizeof(C.PyPtr)
            return UnsafePyObject
        else
            error("object pointer of this size not supported: $size")
        end
    elseif kind == 83  # S = byte string
        error("byte strings not supported")
    elseif kind == 85  # U = unicode string
        mod(size, 4) == 0 || error("unicode size must be a multiple of 4: $size")
        return Utils.StaticString{UInt32,div(size, 4)}
    elseif kind == 86  # V = void (should have descr)
        hasdescr || error("not supported: void dtype with no descr")
        descr = pynew(incref(src.info.descr))
        T = pyarray_descr_to_type(descr)
        sizeof(T) == size || error("size mismatch: itemsize=$size but sizeof(descr)=$(sizeof(T))")
        return T
    else
        error("not supported: dtype of kind: $(Char(kind))")
    end
    @assert false
end

function pyarray_get_ptr(src::PyArraySource_ArrayStruct, ::Type{R}) where {R}
    return Ptr{R}(src.info.data)
end

function pyarray_get_N(src::PyArraySource_ArrayStruct)
    return Int(src.info.nd)
end

function pyarray_get_size(src::PyArraySource_ArrayStruct, ::Val{N}) where {N}
    ptr = src.info.shape
    return ntuple(i->Int(unsafe_load(ptr, i)), Val(N))
end

function pyarray_get_strides(src::PyArraySource_ArrayStruct, ::Val{N}, ::Type{R}, size::NTuple{N,Int}) where {N,R}
    ptr = src.info.strides
    return ntuple(i->Int(unsafe_load(ptr, i)), Val(N))
end

function pyarray_get_M(src::PyArraySource_ArrayStruct)
    return Utils.isflagset(src.info.flags, C.NPY_ARRAY_WRITEABLE)
end

function pyarray_get_handle(src::PyArraySource_ArrayStruct)
    return src.capsule
end

# buffer protocol

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

const PYARRAY_BUFFERFORMAT_TO_TYPE = let c = Utils.islittleendian() ? '<' : '>'
    Dict(
        "b" => Cchar,
        "B" => Cuchar,
        "h" => Cshort,
        "H" => Cushort,
        "i" => Cint,
        "I" => Cuint,
        "l" => Clong,
        "L" => Culong,
        "q" => Clonglong,
        "Q" => Culonglong,
        "e" => Float16,
        "f" => Cfloat,
        "d" => Cdouble,
        "$(c)b" => Cchar,
        "$(c)B" => Cuchar,
        "$(c)h" => Cshort,
        "$(c)H" => Cushort,
        "$(c)i" => Cint,
        "$(c)I" => Cuint,
        "$(c)l" => Clong,
        "$(c)L" => Culong,
        "$(c)q" => Clonglong,
        "$(c)Q" => Culonglong,
        "$(c)e" => Float16,
        "$(c)f" => Cfloat,
        "$(c)d" => Cdouble,
        "?" => Bool,
        "P" => Ptr{Cvoid},
        "O" => UnsafePyObject,
        "=e" => Float16,
        "=f" => Float32,
        "=d" => Float64,
    )
end

pyarray_bufferformat_to_type(fmt::String) = get(()->error("not implemented: buffer format $(repr(fmt))"), PYARRAY_BUFFERFORMAT_TO_TYPE, fmt)

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

Base.unsafe_convert(::Type{Ptr{T}}, x::PyArray{T,N,M,L,T}) where {T,N,M,L} = x.ptr

Base.strides(x::PyArray{T,N,M,L,R}) where {T,N,M,L,R} =
    if all(mod.(x.strides, sizeof(R)) .== 0)
        div.(x.strides, sizeof(R))
    else
        error("strides are not a multiple of element size")
    end

function Base.showarg(io::IO, x::PyArray{T,N}, toplevel::Bool) where {T, N}
    toplevel || print(io, "::")
    print(io, "PyArray{")
    show(io, T)
    print(io, ", ", N, "}")
    return
end

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

function pyarray_load(::Type{T}, p::Ptr{R}) where {T,R}
    if R == T
        unsafe_load(p)
    elseif R == UnsafePyObject
        u = unsafe_load(p)
        o = u.ptr == C_NULL ? pynew(Py(nothing)) : pynew(incref(u.ptr))
        T == Py ? o : pyconvert(T, o)
    else
        convert(T, unsafe_load(p))
    end
end

function pyarray_store!(p::Ptr{R}, x::T) where {R,T}
    if R == T
        unsafe_store!(p, x)
    elseif R == UnsafePyObject
        @autopy x begin
            decref(unsafe_load(p).ptr)
            unsafe_store!(p, UnsafePyObject(incref(getptr(x_))))
        end
    else
        unsafe_store!(p, convert(R, x))
    end
end

function pyarray_get_T(::Type{R}, ::Type{T0}, ::Type{T1}) where {R,T0,T1}
    if R == UnsafePyObject
        if T0 <: Py <: T1
            Py
        else
            T1
        end
    elseif T0 <: R <: T1
        R
    else
        error("impossible")
    end
end

function pyarray_check_T(::Type{T}, ::Type{R}) where {T,R}
    if R == UnsafePyObject
        nothing
    elseif T == R
        nothing
    elseif T <: Number && R <: Number
        nothing
    elseif T <: AbstractString && R <: AbstractString
        nothing
    else
        error("invalid eltype T=$T for raw eltype R=$R")
    end
end   
