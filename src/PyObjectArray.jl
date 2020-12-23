"""
    PyObjectArray(undef, dims...)
    PyObjectArray(array)

An array of `PyObject`s which supports the Python buffer protocol.

Internally, the objects are stored as an array of pointers.
"""
mutable struct PyObjectArray{N} <: AbstractArray{PyObject, N}
    ptrs :: Array{CPyPtr, N}
    function PyObjectArray{N}(::UndefInitializer, dims::NTuple{N,Integer}) where {N}
        x = new{N}(fill(CPyPtr(C_NULL), dims))
        finalizer(x) do x
            if CONFIG.isinitialized
                with_gil(false) do
                    for ptr in x.ptrs
                        C.Py_DecRef(ptr)
                    end
                end
            end
        end
    end
end
PyObjectArray{N}(::UndefInitializer, dims::Vararg{Integer,N}) where {N} = PyObjectArray(undef, dims)
PyObjectArray(::UndefInitializer, dims::NTuple{N,Integer}) where {N} = PyObjectArray{N}(undef, dims)
PyObjectArray(::UndefInitializer, dims::Vararg{Integer,N}) where {N} = PyObjectArray{N}(undef, dims)
PyObjectArray{N}(x::AbstractArray{T,N}) where {T,N} = copy!(PyObjectArray{N}(undef, size(x)), x)
PyObjectArray(x::AbstractArray{T,N}) where {T,N} = PyObjectArray{N}(x)
export PyObjectArray

Base.IndexStyle(x) = Base.IndexStyle(x.ptrs)

Base.length(x::PyObjectArray) = length(x.ptrs)

Base.size(x::PyObjectArray) = size(x.ptrs)

Base.isassigned(x::PyObjectArray, i::Vararg{Integer}) = checkbounds(Bool, x.ptrs, i...) && x.ptrs[i...] != C_NULL

function Base.getindex(x::PyObjectArray, i::Integer...)
    ptr = x.ptrs[i...]
    ptr == C_NULL ? throw(UndefRefError()) : pyborrowedobject(ptr)
end

function Base.setindex!(x::PyObjectArray, _v, i::Integer...)
    vo = C.PyObject_From(_v)
    isnull(vo) && pythrow()
    C.Py_DecRef(x.ptrs[i...])
    C.Py_IncRef(vo)
    x.ptrs[i...] = vo
    x
end

function Base.deleteat!(x::PyObjectArray, i::Integer)
    C.Py_DecRef(x.ptrs[i])
    deleteat!(x.ptrs, i)
    x
end

C._pyjlarray_get_buffer(o, buf, flags, x::PyObjectArray) =
    C.pyjl_get_buffer_impl(o, buf, flags, pointer(x.ptrs), sizeof(CPyPtr), length(x), ndims(x), "O", size(x), strides(x.ptrs), true)

C._pyjlarray_array_interface(x::PyObjectArray) =
    C.PyDict_From(Dict(
        "shape" => size(x),
        "typestr" => "|O",
        "data" => (UInt(pointer(x.ptrs)), false),
        "strides" => strides(x.ptrs) .* sizeof(CPyPtr),
        "version" => 3,
    ))
