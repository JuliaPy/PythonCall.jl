PyObjectArray{N}(undef::UndefInitializer, dims::Vararg{Integer,N}) where {N} =
    PyObjectArray(undef, dims)
PyObjectArray(undef::UndefInitializer, dims::NTuple{N,Integer}) where {N} =
    PyObjectArray{N}(undef, dims)
PyObjectArray(undef::UndefInitializer, dims::Vararg{Integer,N}) where {N} =
    PyObjectArray{N}(undef, dims)
PyObjectArray{N}(x::AbstractArray{T,N}) where {T,N} =
    copyto!(PyObjectArray{N}(undef, size(x)), x)
PyObjectArray(x::AbstractArray{T,N}) where {T,N} = PyObjectArray{N}(x)

pyobjectarray_finalizer(x::PyObjectArray) = GC.enqueue_all(x.ptrs)

Base.IndexStyle(x::PyObjectArray) = Base.IndexStyle(x.ptrs)

Base.length(x::PyObjectArray) = length(x.ptrs)

Base.size(x::PyObjectArray) = size(x.ptrs)

@propagate_inbounds function Base.isassigned(x::PyObjectArray, i::Vararg{Integer})
    @boundscheck checkbounds(Bool, x, i...) || return false
    return @inbounds x.ptrs[i...] != C_NULL
end

@propagate_inbounds function Base.getindex(x::PyObjectArray, i::Integer...)
    @boundscheck checkbounds(x, i...)
    Base.GC.@preserve x begin
        @inbounds ptr = x.ptrs[i...]
        ptr == C_NULL && throw(UndefRefError())
        return pynew(incref(C.PyPtr(ptr)))
    end
end

@propagate_inbounds function Base.setindex!(x::PyObjectArray, v, i::Integer...)
    @boundscheck checkbounds(x, i...)
    v_ = Py(v)
    @inbounds decref(C.PyPtr(x.ptrs[i...]))
    @inbounds x.ptrs[i...] = getptr(incref(v_))
    return x
end

@propagate_inbounds function Base.deleteat!(x::PyObjectVector, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds decref(C.PyPtr(x.ptrs[i]))
    deleteat!(x.ptrs, i)
    return x
end

function pyjlarray_array_interface(x::PyObjectArray)
    return pydict(
        shape = size(x),
        typestr = "|O",
        data = (UInt(pointer(x.ptrs)), false),
        strides = strides(x.ptrs) .* sizeof(C.PyPtr),
        version = 3,
    )
end

# C._pyjlarray_get_buffer(o, buf, flags, x::PyObjectArray) = C.pyjl_get_buffer_impl(
#     o,
#     buf,
#     flags,
#     pointer(x.ptrs),
#     sizeof(CPyPtr),
#     length(x),
#     ndims(x),
#     "O",
#     size(x),
#     strides(x.ptrs) .* sizeof(CPyPtr),
#     true,
# )
