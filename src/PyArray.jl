"""
    PyArray{T,N,R,M,L}(o)

Interpret the Python array `o` as a Julia array.

The input may be anything supporting the buffer protocol or the numpy array interface.
This includes `bytes`, `bytearray`, `array.array`, `numpy.ndarray`, `pandas.Series`.

All type parameters are optional:
- `T` is the (Julia) element type.
- `N` is the number of dimensions.
- `R` is the type of elements of the underlying buffer (which may be different from `T` to allow some basic conversion).
- `M` is true if the array is mutable.
- `L` is true if the array supports fast linear indexing.
"""
mutable struct PyArray{T,N,R,M,L} <: AbstractArray{T,N}
    ref :: PyRef
    ptr :: Ptr{R}
    size :: NTuple{N,Int}
    length :: Int
    bytestrides :: NTuple{N,Int}
    handle :: Any
end
const PyVector{T,R,M,L} = PyArray{T,1,R,M,L}
const PyMatrix{T,R,M,L} = PyArray{T,2,R,M,L}
export PyArray, PyVector, PyMatrix

ispyreftype(::Type{<:PyArray}) = true
pyptr(x::PyArray) = pyptr(x.ref)
Base.unsafe_convert(::Type{CPyPtr}, x::PyArray) = pyptr(x.ref)
C.PyObject_TryConvert__initial(o, ::Type{T}) where {T<:PyArray} = CTryConvertRule_trywrapref(o, T)

function PyArray{T,N,R,M,L}(o::PyRef, info) where {T,N,R,M,L}
    # T - array element type
    T isa Type || error("T must be a type, got T=$T")

    # N - number of dimensions
    N isa Integer || error("N must be an integer, got N=$N")
    N isa Int || return PyArray{T, Int(N), R, M, L}(o, info)
    N == info.ndims || error("source dimension is $(info.ndims), got N=$N")

    # R - buffer element type
    R isa Type || error("R must be a type, got R=$R")
    Base.allocatedinline(R) || error("source elements must be allocated inline, got R=$R")
    Base.aligned_sizeof(R) == info.elsize || error("source elements must have size $(info.elsize), got R=$R")

    # M - mutable
    M isa Bool || error("M must be true or false, got M=$M")
    !M || info.mutable || error("source is immutable, got M=$M")

    bytestrides = info.bytestrides
    size = info.size

    # L - linear indexable
    L isa Bool || error("L must be true or false, got L=$L")
    !L || N ≤ 1 || size_to_fstrides(bytestrides[1], size...) == bytestrides || error("not linearly indexable, got L=$L")

    PyArray{T, N, R, M, L}(PyRef(o), Ptr{R}(info.ptr), size, N==0 ? 1 : prod(size), bytestrides, info.handle)
end
PyArray{T,N,R,M}(o::PyRef, info) where {T,N,R,M} = PyArray{T,N,R,M, N≤1 || size_to_fstrides(info.bytestrides[1], info.size...) == info.bytestrides}(o, info)
PyArray{T,N,R}(o::PyRef, info) where {T,N,R} = PyArray{T,N,R, info.mutable}(o, info)
PyArray{T,N}(o::PyRef, info) where {T,N} = PyArray{T,N, info.eltype}(o, info)
PyArray{T}(o::PyRef, info) where {T} = PyArray{T, info.ndims}(o, info)
PyArray{<:Any,N}(o::PyRef, info) where {N} = PyArray{pyarray_default_T(info.eltype), N}(o, info)
PyArray(o::PyRef, info) = PyArray{pyarray_default_T(info.eltype)}(o, info)

(::Type{A})(o; opts...) where {A<:PyArray} = begin
    ref = PyRef(o)
    info = pyarray_info(ref; opts...)
    info = (
        ndims = Int(info.ndims),
        eltype = info.eltype :: Type,
        elsize = Int(info.elsize),
        mutable = info.mutable :: Bool,
        bytestrides = NTuple{Int(info.ndims), Int}(info.bytestrides),
        size = NTuple{Int(info.ndims), Int}(info.size),
        ptr = Ptr{Cvoid}(info.ptr),
        handle = info.handle,
    )
    A(ref, info)
end

function pyarray_info(ref; buffer=true, array=true, copy=true)
    if array && pyhasattr(ref, "__array_interface__")
        pyconvertdescr(x) = begin
            @py ```
            def convert(x):
                def fix(x):
                    a = x[0]
                    a = (a, a) if isinstance(a, str) else (a[0], a[1])
                    b = x[1]
                    c = x[2] if len(x)>2 else 1
                    return (a, b, c)
                if x is None or isinstance(x, str):
                    return x
                else:
                    return [fix(y) for y in x]
            $(r::Union{Nothing,String,Vector{Tuple{Tuple{String,String}, PyObject, Int}}}) = convert($x)
            ```
            r isa Vector ? [(a, pyconvertdescr(b), c) for (a,b,c) in r] : r
        end
        ai = pygetattr(ref, "__array_interface__")
        pyconvert(Int, ai["version"]) == 3 || error("wrong version")
        size = pyconvert(Tuple{Vararg{Int}}, ai["shape"])
        ndims = length(size)
        typestr = pyconvert(String, ai["typestr"])
        descr = pyconvertdescr(ai.get("descr"))
        eltype = pytypestrdescr_to_type(typestr, descr)
        elsize = Base.aligned_sizeof(eltype)
        strides = pyconvert(Union{Nothing, Tuple{Vararg{Int}}}, ai.get("strides"))
        strides === nothing && (strides = size_to_cstrides(elsize, size...))
        pyis(ai.get("mask"), pynone()) || error("mask not supported")
        offset = pyconvert(Union{Nothing, Int}, ai.get("offset"))
        offset === nothing && (offset = 0)
        data = pyconvert(Union{PyObject, Tuple{UInt, Bool}, Nothing}, ai.get("data"))
        if data isa Tuple
            ptr = Ptr{Cvoid}(data[1])
            mutable = !data[2]
            handle = (ref, ai)
        else
            buf = PyBuffer(data === nothing ? ref : data)
            ptr = buf.buf
            mutable = !buf.readonly
            handle = (ref, ai, buf)
        end
        return (ndims=ndims, eltype=eltype, elsize=elsize, mutable=mutable, bytestrides=strides, size=size, ptr=ptr, handle=handle)
    end
    if array && pyhasattr(ref, "__array_struct__")
        # TODO
    end
    if buffer && C.PyObject_CheckBuffer(ref)
        try
            b = PyBuffer(ref, C.PyBUF_RECORDS_RO)
            return (ndims=b.ndim, eltype=b.eltype, elsize=b.itemsize, mutable=!b.readonly, bytestrides=b.strides, size=b.shape, ptr=b.buf, handle=b)
        catch
        end
    end
    if array && copy && pyhasattr(ref, "__array__")
        try
            return pyarray_info(pycall(PyRef, pygetattr(PyRef, ref, "__array__")); buffer=buffer, array=array, copy=false)
        catch
        end
    end
    error("given object does not support the buffer protocol or array interface")
end

Base.isimmutable(x::PyArray{T,N,R,M,L}) where {T,N,R,M,L} = !M
Base.size(x::PyArray) = x.size
Base.length(x::PyArray) = x.length
Base.IndexStyle(::Type{PyArray{T,N,R,M,L}}) where {T,N,R,M,L} = L ? Base.IndexLinear() : Base.IndexCartesian()

Base.@propagate_inbounds Base.getindex(x::PyArray{T,N,R,M,L}, i::Vararg{Int,N2}) where {T,N,R,M,L,N2} =
    if (N2==N) || (L && N2==1)
        @boundscheck checkbounds(x, i...)
        pyarray_load(T, x.ptr + pyarray_offset(x, i...))
    else
        invoke(getindex, Tuple{AbstractArray{T,N}, Vararg{Int,N2}}, x, i...)
    end

Base.@propagate_inbounds Base.setindex!(x::PyArray{T,N,R,true,L}, v, i::Vararg{Int,N2}) where {T,N,R,L,N2} =
    if (N2==N) || (L && N2==1)
        @boundscheck checkbounds(x, i...)
        pyarray_store!(x.ptr + pyarray_offset(x, i...), convert(T, v))
        x
    else
        invoke(setindex!, Tuple{AbstractArray{T,N}, typeof(v), Vararg{Int,N2}}, x, v, i...)
    end

pyarray_default_T(::Type{R}) where {R} = R
pyarray_default_T(::Type{C.PyObjectRef}) = PyObject

pyarray_load(::Type{T}, p::Ptr{T}) where {T} = unsafe_load(p)
pyarray_load(::Type{T}, p::Ptr{C.PyObjectRef}) where {T} = begin
    o = unsafe_load(p).ptr
    isnull(o) && throw(UndefRefError())
    ism1(C.PyObject_Convert(o, T)) && pythrow()
    takeresult(T)
end

pyarray_store!(p::Ptr{T}, v::T) where {T} = unsafe_store!(p, v)
pyarray_store!(p::Ptr{C.PyObjectRef}, v::T) where {T} = begin
    o = C.PyObject_From(v)
    isnull(o) && pythrow()
    C.Py_DecRef(unsafe_load(p).ptr)
    unsafe_store!(p, C.PyObjectRef(o))
end

pyarray_offset(x::PyArray{T,N,R,M,true}, i::Int) where {T,N,R,M} =
    N==0 ? 0 : (i-1) * x.bytestrides[1]

pyarray_offset(x::PyArray{T,1,R,M,true}, i::Int) where {T,R,M} =
    (i-1) .* x.bytestrides[1]

pyarray_offset(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} =
    sum((i .- 1) .* x.bytestrides)
