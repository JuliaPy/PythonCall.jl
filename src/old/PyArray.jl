"""
    PyArray{T,N,R,M,L}(o)

Interpret the Python array `o` as a Julia array.

Type parameters which are not given or set to `missing` are inferred:
- `T` is the (Julia) element type.
- `N` is the number of dimensions.
- `R` is the type of elements of the underlying buffer (which may be different from `T` to allow some basic conversion).
- `M` is true if the array is mutable.
- `L` is true if the array supports fast linear indexing.
"""
mutable struct PyArray{T,N,R,M,L} <: AbstractArray{T,N}
    o :: PyObject
    ptr :: Ptr{R}
    size :: NTuple{N,Int}
    length :: Int
    bytestrides :: NTuple{N,Int}
    handle :: Any
end

const PyVector{T,R,M,L} = PyArray{T,1,R,M,L}
const PyMatrix{T,R,M,L} = PyArray{T,2,R,M,L}
export PyArray, PyVector, PyMatrix

function PyArray{T,N,R,M,L}(o::PyObject, info=pyarray_info(o)) where {T,N,R,M,L}
    # R - buffer element type
    if R === missing
        return PyArray{T, N, info.eltype, M, L}(o, info)
    elseif R isa Type
        Base.allocatedinline(R) || error("source must be allocated inline, got R=$R")
        Base.aligned_sizeof(R) == info.elsize || error("source elements must have size $(info.elsize), got R=$R")
    else
        error("R must be missing or a type")
    end

    # T - array element type
    if T === missing
        return PyArray{pyarray_default_T(R), N, R, M, L}(o, info)
    elseif T isa Type
        # great
    else
        error("T must be missing or a type")
    end

    # N
    if N === missing
        return PyArray{T, Int(info.ndims), R, M, L}(o, info)
    elseif N isa Int
        N == info.ndims || error("source dimension is $(info.ndims), got N=$N")
        # great
    elseif N isa Integer
        return PyArray{T, Int(N), R, M, L}(o, info)
    else
        error("N must be missing or an integer")
    end

    # M
    if M === missing
        return PyArray{T, N, R, Bool(info.mutable), L}(o, info)
    elseif M === true
        info.mutable || error("source is immutable, got L=$L")
    elseif M === false
        # great
    else
        error("M must be missing, true or false")
    end

    bytestrides = NTuple{N, Int}(info.bytestrides)
    size = NTuple{N, Int}(info.size)

    # L
    if L === missing
        return PyArray{T, N, R, M, N ≤ 1 || size_to_fstrides(bytestrides[1], size...) == bytestrides}(o, info)
    elseif L === true
        N ≤ 1 || size_to_fstrides(bytestrides[1], size...) == bytestrides || error("not linearly indexable")
    elseif L === false
        # great
    else
        error("L must be missing, true or false")
    end

    PyArray{T, N, R, M, L}(o, Ptr{R}(info.ptr), size, N==0 ? 1 : prod(size), bytestrides, info.handle)
end
PyArray{T,N,R,M}(o) where {T,N,R,M} = PyArray{T,N,R,M,missing}(o)
PyArray{T,N,R}(o) where {T,N,R} = PyArray{T,N,R,missing}(o)
PyArray{T,N}(o) where {T,N} = PyArray{T,N,missing}(o)
PyArray{T}(o) where {T} = PyArray{T,missing}(o)
PyArray(o) where {} = PyArray{missing}(o)

pyobject(x::PyArray) = x.o

function pyarray_info(o::PyObject)
    # TODO: support the numpy array interface too
    b = PyBuffer(o, C.PyBUF_RECORDS_RO)
    (ndims=b.ndim, eltype=b.eltype, elsize=b.itemsize, mutable=!b.readonly, bytestrides=b.strides, size=b.shape, ptr=b.buf, handle=b)
end

Base.isimmutable(x::PyArray{T,N,R,M,L}) where {T,N,R,M,L} = !M
Base.pointer(x::PyArray{T,N,T}) where {T,N} = x.ptr
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
pyarray_default_T(::Type{CPyObjRef}) = PyObject

pyarray_load(::Type{T}, p::Ptr{T}) where {T} = unsafe_load(p)
pyarray_load(::Type{T}, p::Ptr{CPyObjRef}) where {T} = (o=unsafe_load(p).ptr; o==C_NULL ? throw(UndefRefError()) : pyconvert(T, pyborrowedobject(o)))

pyarray_store!(p::Ptr{T}, v::T) where {T} = unsafe_store!(p, v)
pyarray_store!(p::Ptr{CPyObjRef}, v::T) where {T} = (C.Py_DecRef(unsafe_load(p).ptr); unsafe_store!(p, CPyObjRef(pyptr(pyincref!(pyobject(v))))))

pyarray_offset(x::PyArray{T,N,R,M,true}, i::Int) where {T,N,R,M} =
    N==0 ? 0 : (i-1) * x.bytestrides[1]

pyarray_offset(x::PyArray{T,1,R,M,true}, i::Int) where {T,R,M} =
    (i-1) .* x.bytestrides[1]

pyarray_offset(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} =
    sum((i .- 1) .* x.bytestrides)
