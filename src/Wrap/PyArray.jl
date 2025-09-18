ispy(::PyArray) = true
Py(x::PyArray) = x.py

(::Type{A})(
    x;
    array::Bool = true,
    buffer::Bool = true,
    copy::Bool = true,
) where {A<:PyArray} = @autopy x begin
    r = pyarray_make(A, x_, array = array, buffer = buffer, copy = copy)
    if pyconvert_isunconverted(r)
        error("cannot convert this Python '$(pytype(x_).__name__)' to a '$A'")
    else
        return pyconvert_result(r)::A
    end
end

pyconvert_rule_array_nocopy(::Type{A}, x::Py) where {A<:PyArray} =
    pyarray_make(A, x, copy = false)

function pyconvert_rule_array(::Type{A}, x::Py) where {A<:AbstractArray}
    r = pyarray_make(PyArray, x)
    if pyconvert_isunconverted(r)
        return pyconvert_unconverted()
    else
        return pyconvert_tryconvert(A, pyconvert_result(PyArray, r))
    end
end

abstract type PyArraySource end

function pyarray_make(
    ::Type{A},
    x::Py;
    array::Bool = true,
    buffer::Bool = true,
    copy::Bool = true,
) where {A<:PyArray}
    # TODO: try/catch is SLOW if an error is thrown, think about sending errors via return values instead
    A == Union{} && return pyconvert_unconverted()
    if array && (xa = pygetattr(x, "__array_struct__", PyNULL); !pyisnull(xa))
        try
            return pyarray_make(A, x, PyArraySource_ArrayStruct(x, xa))
        catch exc
            @debug "failed to make PyArray from __array_struct__" exc = exc
        end
    end
    if array && (xi = pygetattr(x, "__array_interface__", PyNULL); !pyisnull(xi))
        try
            return pyarray_make(A, x, PyArraySource_ArrayInterface(x, xi))
        catch exc
            @debug "failed to make PyArray from __array_interface__" exc = exc
        end
    end
    if buffer && C.PyObject_CheckBuffer(x)
        try
            return pyarray_make(A, x, PyArraySource_Buffer(x))
        catch exc
            @debug "failed to make PyArray from buffer" exc = exc
        end
    end
    if copy && array && pyhasattr(x, "__array__")
        y = x.__array__()
        if (ya = pygetattr(y, "__array_struct__", PyNULL); !pyisnull(ya))
            try
                return pyarray_make(A, y, PyArraySource_ArrayStruct(y, ya))
            catch exc
                @debug "failed to make PyArray from __array__().__array_interface__" exc =
                    exc
            end
        end
        if (yi = pygetattr(y, "__array_interface__", PyNULL); !pyisnull(yi))
            try
                return pyarray_make(A, y, PyArraySource_ArrayInterface(y, yi))
            catch exc
                @debug "failed to make PyArray from __array__().__array_interface__" exc =
                    exc
            end
        end
    end
    return pyconvert_unconverted()
end

function pyarray_make(
    ::Type{A},
    x::Py,
    info::PyArraySource,
    ::Type{PyArray{T0,N0,F0}} = Utils._type_lb(A),
    ::Type{PyArray{T1,N1,F1}} = Utils._type_ub(A),
) where {A<:PyArray,T0,N0,F0,T1,N1,F1}
    # T (eltype) - checked against R (ptr eltype)
    R′ = pyarray_get_R(info)::DataType
    if T0 == T1
        T = T1
        R = pyarray_get_R(T)
        R′ == R || error("eltype $T is incompatible with array data of type $R′")
    else
        T = pyarray_get_T(R′, T0, T1)::DataType
        T <: T1 || error("computed eltype $T out of bounds, should be <: $T1")
        T >: T0 || error("computed eltype $T out of bounds, should be >: $T0")
        R = pyarray_get_R(T)
        R′ == R || error("computed eltype $T is incompatible with array data of type $R′")
    end
    # ptr
    ptr = pyarray_get_ptr(info)::Ptr{Cvoid}
    # N (ndims)
    N′ = pyarray_get_N(info)::Int
    if N0 == N1
        N = N1 isa Int ? N1 : Int(N1)
        N == N′ || error("number of dimensions $N incorrect, expecting $N′")
    else
        N = N′
    end
    # size
    size = pyarray_get_size(info, Val(N))::NTuple{N,Int}
    # strides
    strides = pyarray_get_strides(info, Val(N), R, size)::NTuple{N,Int}
    # F (flags)
    M = pyarray_get_M(info)::Bool
    L = N < 2 || strides == Utils.size_to_fstrides(strides[1], size)
    C = L && (N == 0 || strides[1] == sizeof(R))
    Flist = Symbol[]
    if F0 == F1
        F = F1::Tuple{Vararg{Symbol}}
        (:mutable in F) && (!M) && error(":mutable flag given but array is not mutable")
        (:linear in F) && (!L) && error(":linear flag given but array is not evenly spaced")
        (:contiguous in F) &&
            (!C) &&
            error(":contiguous flag given but array is not contiguous")
    else
        M && push!(Flist, :mutable)
        L && push!(Flist, :linear)
        C && push!(Flist, :contiguous)
        F = Tuple(Flist)
    end
    # handle
    handle = pyarray_get_handle(info)
    # done
    arr = PyArray{T,N,F}(Val(:new), ptr, size, strides, x, handle)
    return pyconvert_return(arr)
end

# array interface

struct PyArraySource_ArrayInterface <: PyArraySource
    obj::Py
    dict::Py
    ptr::Ptr{Cvoid}
    readonly::Bool
    handle::Py
end
function PyArraySource_ArrayInterface(x::Py, d::Py = x.__array_interface__)
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
        buf = UnsafePtr(C.PyMemoryView_GET_BUFFER(memview))
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
        sizeof(T) == sz ||
            error("size mismatch: itemsize=$sz but sizeof(descr)=$(sizeof(T))")
        return T
    elseif etc == 'M' || etc == 'm'
        m = match(r"^([0-9]+)(\[([0-9]+)?([a-zA-Z]+)\])?$", ts[3:end])
        m === nothing && error("could not parse type: $ts")
        sz = parse(Int, m[1])
        sz == sizeof(Int64) || error(
            "$(etc == 'M' ? "datetime" : "timedelta") of this size not supported: $sz",
        )
        s = m[3] === nothing ? 1 : parse(Int, m[3])
        us = m[4] === nothing ? "" : m[4]
        u =
            us == "" ? NumpyDates.UNBOUND_UNITS :
            us == "Y" ? NumpyDates.YEARS :
            us == "M" ? NumpyDates.MONTHS :
            us == "W" ? NumpyDates.WEEKS :
            us == "D" ? NumpyDates.DAYS :
            us == "h" ? NumpyDates.HOURS :
            us == "m" ? NumpyDates.MINUTES :
            us == "s" ? NumpyDates.SECONDS :
            us == "ms" ? NumpyDates.MILLISECONDS :
            us == "us" ? NumpyDates.MICROSECONDS :
            us == "ns" ? NumpyDates.NANOSECONDS :
            us == "ps" ? NumpyDates.PICOSECONDS :
            us == "fs" ? NumpyDates.FEMTOSECONDS :
            us == "as" ? NumpyDates.ATTOSECONDS :
            error("not supported: $(etc == 'M' ? "datetime" : "timedelta") unit: $us")
        if etc == 'M'
            return NumpyDates.InlineDateTime64{s == 1 ? u : (u, s)}
        else
            return NumpyDates.InlineTimeDelta64{s == 1 ? u : (u, s)}
        end
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
    if fnames == [Symbol(:f, i - 1) for i = 1:length(fnames)]
        return T
    else
        return NamedTuple{Tuple(fnames),T}
    end
end

function pyarray_get_R(src::PyArraySource_ArrayInterface)
    typestr = pyconvert(String, src.dict["typestr"])
    descr = @py @jl(src.dict).get("descr")
    R = pyarray_typestrdescr_to_type(typestr, descr)::DataType
    pydel!(descr)
    return R
end

pyarray_get_ptr(src::PyArraySource_ArrayInterface) = src.ptr

pyarray_get_N(src::PyArraySource_ArrayInterface) = Int(@py jllen(@jl(src.dict)["shape"]))

pyarray_get_size(src::PyArraySource_ArrayInterface, ::Val{N}) where {N} =
    pyconvert(NTuple{N,Int}, src.dict["shape"])

function pyarray_get_strides(
    src::PyArraySource_ArrayInterface,
    ::Val{N},
    ::Type{R},
    size::NTuple{N,Int},
) where {R,N}
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
    obj::Py
    capsule::Py
    info::C.PyArrayInterface
end
function PyArraySource_ArrayStruct(x::Py, capsule::Py = x.__array_struct__)
    name = C.PyCapsule_GetName(capsule)
    ptr = C.PyCapsule_GetPointer(capsule, name)
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
        error("datetime not supported")
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
        sizeof(T) == size ||
            error("size mismatch: itemsize=$size but sizeof(descr)=$(sizeof(T))")
        return T
    else
        error("not supported: dtype of kind: $(Char(kind))")
    end
    @assert false
end

function pyarray_get_ptr(src::PyArraySource_ArrayStruct)
    src.info.data
end

function pyarray_get_N(src::PyArraySource_ArrayStruct)
    return Int(src.info.nd)
end

function pyarray_get_size(src::PyArraySource_ArrayStruct, ::Val{N}) where {N}
    ptr = src.info.shape
    return ntuple(i -> Int(unsafe_load(ptr, i)), Val(N))
end

function pyarray_get_strides(
    src::PyArraySource_ArrayStruct,
    ::Val{N},
    ::Type{R},
    size::NTuple{N,Int},
) where {N,R}
    ptr = src.info.strides
    return ntuple(i -> Int(unsafe_load(ptr, i)), Val(N))
end

function pyarray_get_M(src::PyArraySource_ArrayStruct)
    return Utils.isflagset(src.info.flags, C.NPY_ARRAY_WRITEABLE)
end

function pyarray_get_handle(src::PyArraySource_ArrayStruct)
    return src.capsule
end

# buffer protocol

struct PyArraySource_Buffer <: PyArraySource
    obj::Py
    memview::Py
    buf::C.UnsafePtr{C.Py_buffer}
end
function PyArraySource_Buffer(x::Py)
    memview = pybuiltins.memoryview(x)
    buf = C.UnsafePtr(C.PyMemoryView_GET_BUFFER(memview))
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
        "O" => C.PyPtr,
        "=e" => Float16,
        "=f" => Float32,
        "=d" => Float64,
    )
end

pyarray_bufferformat_to_type(fmt::String) = get(
    () -> error("not implemented: buffer format $(repr(fmt))"),
    PYARRAY_BUFFERFORMAT_TO_TYPE,
    fmt,
)

function pyarray_get_R(src::PyArraySource_Buffer)
    ptr = src.buf.format[]
    return ptr == C_NULL ? UInt8 : pyarray_bufferformat_to_type(String(ptr))
end

pyarray_get_ptr(src::PyArraySource_Buffer) = src.buf.buf[!]

pyarray_get_N(src::PyArraySource_Buffer) = Int(src.buf.ndim[])

function pyarray_get_size(src::PyArraySource_Buffer, ::Val{N}) where {N}
    size = src.buf.shape[]
    if size == C_NULL
        N == 0 ? () : N == 1 ? (Int(src.buf.len[]),) : @assert false
    else
        ntuple(i -> Int(size[i]), N)
    end
end

function pyarray_get_strides(
    src::PyArraySource_Buffer,
    ::Val{N},
    ::Type{R},
    size::NTuple{N,Int},
) where {N,R}
    strides = src.buf.strides[]
    if strides == C_NULL
        itemsize = src.buf.shape[] == C_NULL ? 1 : src.buf.itemsize[]
        Utils.size_to_cstrides(itemsize, size)
    else
        ntuple(i -> Int(strides[i]), N)
    end
end

pyarray_get_M(src::PyArraySource_Buffer) = src.buf.readonly[] == 0

pyarray_get_handle(src::PyArraySource_Buffer) = src.memview

# AbstractArray methods

Base.length(x::PyArray) = x.length

Base.size(x::PyArray) = x.size

Utils.ismutablearray(x::PyArray{T,N,F}) where {T,N,F} = (:mutable in F)

Base.IndexStyle(::Type{PyArray{T,N,F}}) where {T,N,F} =
    (:linear in F) ? Base.IndexLinear() : Base.IndexCartesian()

Base.unsafe_convert(::Type{Ptr{T}}, x::PyArray{T}) where {T} =
    pyarray_get_R(T) == T ? Ptr{T}(x.ptr) : error("")

Base.elsize(::Type{PyArray{T}}) where {T} = pyarray_get_R(T) == T ? sizeof(T) : error("")

function Base.strides(x::PyArray{T}) where {T}
    R = pyarray_get_R(T)
    if all(mod.(x.strides, sizeof(R)) .== 0)
        div.(x.strides, sizeof(R))
    else
        error("strides are not a multiple of element size")
    end
end

@propagate_inbounds Base.getindex(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} =
    pyarray_getindex(x, i...)
@propagate_inbounds Base.getindex(x::PyArray, i::Int) = pyarray_getindex(x, i)

@propagate_inbounds function pyarray_getindex(x::PyArray{T}, i...) where {T}
    @boundscheck checkbounds(x, i...)
    R = pyarray_get_R(T)
    pyarray_load(T, Ptr{R}(x.ptr + pyarray_offset(x, i...)))
end

@propagate_inbounds Base.setindex!(x::PyArray{T,N}, v, i::Vararg{Int,N}) where {T,N} =
    pyarray_setindex!(x, v, i...)
@propagate_inbounds Base.setindex!(x::PyArray, v, i::Int) = pyarray_setindex!(x, v, i)

@propagate_inbounds function pyarray_setindex!(x::PyArray{T,N,F}, v, i...) where {T,N,F}
    (:mutable in F) || error("array is immutable")
    @boundscheck checkbounds(x, i...)
    R = pyarray_get_R(T)
    pyarray_store!(T, Ptr{R}(x.ptr + pyarray_offset(x, i...)), convert(T, v)::T)
    x
end

function pyarray_offset(x::PyArray{T,N,F}, i::Int) where {T,N,F}
    if N == 0
        0
    elseif (:contiguous in F)
        (i - 1) * sizeof(pyarray_get_R(T))
    elseif (N == 1) || (:linear in F)
        (i - 1) * x.strides[1]
    else
        # convert i to cartesian indices
        # there's no public function for this :(
        j = Base._to_subscript_indices(x, i)
        sum((j .- 1) .* x.strides)
    end
end

function pyarray_offset(x::PyArray{T,N,F}, i::Vararg{Int,N}) where {T,N,F}
    sum((i .- 1) .* x.strides)
end

function pyarray_load(::Type{T}, p::Ptr{R}) where {T,R}
    if R == T
        unsafe_load(p)
    elseif R == C.PyPtr
        u = unsafe_load(p)
        o = u == C_NULL ? pynew(Py(nothing)) : pynew(incref(u))
        T == Py ? o : pyconvert(T, o)
    else
        convert(T, unsafe_load(p))
    end
end

function pyarray_store!(::Type{T}, p::Ptr{R}, x::T) where {R,T}
    if R == T
        unsafe_store!(p, x)
    elseif R == C.PyPtr
        @autopy x begin
            decref(unsafe_load(p).ptr)
            unsafe_store!(p, getptr(incref(x_)))
        end
    else
        unsafe_store!(p, convert(R, x))
    end
end

@generated function pyarray_get_T(::Type{R}, ::Type{T0}, ::Type{T1}) where {R,T0,T1}
    if R == C.PyPtr
        if T0 <: Py <: T1
            Py
        else
            error("impossible")
        end
    elseif R <: Tuple
        error("not implemented")
    elseif T0 <: R <: T1
        R
    else
        error("impossible")
    end
end

@generated function pyarray_get_R(::Type{T}) where {T}
    if (
        T == Bool ||
        T == Int8 ||
        T == Int16 ||
        T == Int32 ||
        T == Int64 ||
        T == UInt8 ||
        T == UInt16 ||
        T == UInt32 ||
        T == UInt64 ||
        T == Float16 ||
        T == Float32 ||
        T == Float64 ||
        T == Complex{Float16} ||
        T == Complex{Float32} ||
        T == Complex{Float16}
    )
        T
    elseif isconcretetype(T) &&
           isbitstype(T) &&
           (
               T <: NumpyDates.InlineDateTime64 ||
               T <: NumpyDates.InlineTimeDelta64 ||
               T <: Ptr
           )
        T
    elseif T == Py
        C.PyPtr
    elseif T <: Tuple
        Tuple{map(pyarray_get_R, T.parameters)...}
    else
        error("unsupported eltype $T")
    end
end
