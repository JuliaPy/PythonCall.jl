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
    ptr :: Ptr{R}
    size :: NTuple{N,Int}
    length :: Int
    bytestrides :: NTuple{N,Int}
    handle :: Any
end
export PyArray

function PyArray(o::AbstractPyObject)
    b = PyBuffer(o)
    PyArray{b.eltype, Int(b.ndim), b.eltype, !b.readonly, b.ndim ≤ 1 || size_to_fstrides(b.strides[1], b.shape...)==b.strides}(Ptr{b.eltype}(b.buf), b.shape, b.ndim==0 ? 1 : prod(b.shape), b.strides, b)
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
        invoke(setindex!, Tuple{AbstractArray{T,N}, T, Vararg{Int,N2}}, x, v, i...)
    end

pyarray_load(::Type{T}, p::Ptr{T}) where {T} = unsafe_load(p)

pyarray_store!(p::Ptr{T}, v::T) where {T} = unsafe_store!(p, v)

pyarray_offset(x::PyArray{T,N,R,M,true}, i::Int) where {T,N,R,M} =
    N==0 ? 0 : (i-1) * x.bytestrides[1]

pyarray_offset(x::PyArray{T,1,R,M,true}, i::Int) where {T,R,M} =
    (i-1) .* x.bytestrides[1]

pyarray_offset(x::PyArray{T,N}, i::Vararg{Int,N}) where {T,N} =
    sum((i .- 1) .* x.bytestrides)
