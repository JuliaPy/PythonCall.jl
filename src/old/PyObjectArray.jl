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
    v = convert(PyObject, _v)
    C.Py_DecRef(x.ptrs[i...])
    pyincref!(v)
    x.ptrs[i...] = pyptr(v)
    x
end

function Base.deleteat!(x::PyObjectArray, i::Integer)
    C.Py_DecRef(x.ptrs[i])
    deleteat!(x.ptrs, i)
    x
end
