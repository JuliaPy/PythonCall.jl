"""
    PyArray{R,N,T,M,L}(o) :: AbstractArray{T,N}

Interpret the Python array `o` as a Julia array.

The input may be anything supporting the buffer protocol or the numpy array interface.
This includes `bytes`, `bytearray`, `array.array`, `numpy.ndarray`, `pandas.Series`.

All type parameters are optional:
- `R` is the type of elements of the underlying buffer.
- `N` is the number of dimensions.
- `T` is the element type.
- `M` is true if the array is mutable.
- `L` is true if the array supports fast linear indexing.

There are alias types with names of the form `Py[Mutable/Immutable/][Linear/Cartesian/][Array/Vector/Matrix]`.
"""
mutable struct PyArray{R,N,T,M,L} <: AbstractArray{T,N}
    ptr::CPyPtr
    buf::Ptr{R}
    size::NTuple{N,Int}
    length::Int
    bytestrides::NTuple{N,Int}
    handle::Any
    PyArray{R,N,T,M,L}(::Val{:new}, ptr::CPyPtr, buf::Ptr{R}, size::NTuple{N,Int}, length::Int, bytestrides::NTuple{N,Int}, handle::Any) where {R,N,T,M,L} =
        finalizer(pyref_finalize!, new{R,N,T,M,L}(ptr, buf, size, length, bytestrides, handle))
end
export PyArray

# aliases
for M in (true, false, missing)
    for L in (true, false, missing)
        for N in (1, 2, missing)
            name = Symbol("Py", M===missing ? "" : M ? "Mutable" : "Immutable", L===missing ? "" : L ? "Linear" : "Cartesian", N===missing ? "Array" : N==1 ? "Vector" : "Matrix")
            name == "PyArray" && continue
            lparams = filter(x->x!==nothing, (:R, N===missing ? :N : nothing, :T, M===missing ? :M : nothing, L===missing ? :L : nothing))
            rparams = (:R, N===missing ? :N : N, :T, M===missing ? :M : M, L===missing ? :L : L)
            @eval const $name{$(lparams...)} = PyArray{$(rparams...)}
            @eval export $name
        end
    end
end

ispyreftype(::Type{<:PyArray}) = true
pyptr(x::PyArray) = x.ptr
Base.unsafe_convert(::Type{CPyPtr}, x::PyArray) = pyptr(x)
C.PyObject_TryConvert__initial(o, ::Type{T}) where {T<:PyArray} =
    CTryConvertRule_trywrapref(o, T)

(::Type{A})(o; opts...) where {A<:PyArray} = begin
    ref = o isa C.PyObjectRef ? PyRef(o) : ispyref(o) ? o : PyRef(o)
    info = pyarray_info(ref; opts...)
    R = pyarray_R(A, info)
    N = pyarray_N(A, info)
    T = pyarray_T(A, info, R)
    M = pyarray_M(A, info)
    L = pyarray_L(A, info, Val(N))
    size = pyarray_size(info, Val(N))
    PyArray{R,N,T,M,L}(
        Val(:new),
        pyptr(ref),
        Ptr{R}(pyarray_ptr(info)),
        size,
        N == 0 ? 1 : prod(size),
        pyarray_bytestrides(info, Val(N)),
        pyarray_handle(info),
    )
end

pyarray_R(::Type{A}, info) where {R, A<:PyArray{R}} =
    !isa(R, Type) ? error("R must be a type, got R=$R") :
    !allocatedinline(R) ? error("source elements must be allocated inline, got R=$R") :
    sizeof(R) != pyarray_elsize(info) ? error("source elements must have size $(pyarray_elsize(info)), got R=$R") : R
pyarray_R(::Type{A}, info) where {A<:PyArray} = pyarray_R(info)::Type

pyarray_N(::Type{A}, info) where {N,A<:(PyArray{R,N} where R)} =
    !isa(N, Int) ? error("N must be an Int") :
    N != pyarray_N(info) ? error("source dimension is $(pyarray_N(info)), got N=$N") : N
pyarray_N(::Type{A}, info) where {A<:PyArray} = pyarray_N(info)::Int

pyarray_T(::Type{A}, info, ::Type{R}) where {T,A<:(PyArray{R,N,T} where {R,N}),R} =
    !isa(T, Type) ? error("T must be a type, got T=$T") : T
pyarray_T(::Type{A}, info, ::Type{R}) where {A<:PyArray,R} = pyarray_default_T(R)

pyarray_M(::Type{A}, info) where {M,A<:(PyArray{R,N,T,M} where {R,N,T})} =
    !isa(M, Bool) ? error("M must be a bool, got M=$M") :
    (M && !pyarray_M(info)) ? error("source is immutable, got M=$M") : M
pyarray_M(::Type{A}, info) where {A<:PyArray} = pyarray_M(info)

pyarray_L(::Type{A}, info, ::Val{N}) where {L,A<:(PyArray{R,N,T,M,L} where {R,N,T,M}),N} =
    !isa(L, Bool) ? error("L must be a bool, got L=$L") :
    (L && !pyarray_L(info, Val(N))) ? error("source is not linearly indexable, got L=$L") : L
pyarray_L(::Type{A}, info, ::Val{N}) where {A<:PyArray, N} = pyarray_L(info, Val(N))

abstract type PyArrayInfo end

pyarray_L(info::PyArrayInfo, ::Val{N}) where {N} = begin
    N ≤ 1 && return true
    strides = pyarray_bytestrides(info, Val(N))
    size = pyarray_size(info, Val(N))
    strides == size_to_fstrides(strides[1], size...)
end

struct PyArrayInfo_Buffer <: PyArrayInfo
    buf :: PyBuffer
end

pyarray_R(info::PyArrayInfo_Buffer) = info.buf.eltype
pyarray_N(info::PyArrayInfo_Buffer) = Int(info.buf.ndim)
pyarray_M(info::PyArrayInfo_Buffer) = !info.buf.readonly
pyarray_ptr(info::PyArrayInfo_Buffer) = info.buf.buf
pyarray_size(info::PyArrayInfo_Buffer, ::Val{N}) where {N} = begin
    p = info.buf.info[].shape
    ntuple(i->Int(unsafe_load(p,i)), Val(N))::NTuple{N,Int}
end
pyarray_bytestrides(info::PyArrayInfo_Buffer, ::Val{N}) where {N} = begin
    p = info.buf.info[].strides
    ntuple(i->Int(unsafe_load(p,i)), Val(N))::NTuple{N,Int}
end
pyarray_handle(info::PyArrayInfo_Buffer) = info.buf
pyarray_elsize(info::PyArrayInfo_Buffer) = info.buf.itemsize

struct PyArrayInfo_ArrayInterface <: PyArrayInfo
    dict :: PyObject
    ptr :: Ptr{Cvoid}
    mutable :: Bool
    handle :: Any
end

pyarray_R(info::PyArrayInfo_ArrayInterface) = begin
    # converts descr to something more Julian
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
        r isa Vector ? [(a, pyconvertdescr(b), c) for (a, b, c) in r] : r
    end
    typestr = @pyv `$(info.dict)["typestr"]`::String
    descr = pyconvertdescr(@pyv `$(info.dict).get("descr")`)
    pytypestrdescr_to_type(typestr, descr)
end
pyarray_N(info::PyArrayInfo_ArrayInterface) = @pyv `len($(info.dict)["shape"])`::Int
pyarray_M(info::PyArrayInfo_ArrayInterface) = info.mutable
pyarray_ptr(info::PyArrayInfo_ArrayInterface) = info.ptr
pyarray_size(info::PyArrayInfo_ArrayInterface, ::Val{N}) where {N} = @pyv `$(info.dict)["shape"]`::NTuple{N,Int}
pyarray_bytestrides(info::PyArrayInfo_ArrayInterface, ::Val{N}) where {N} = begin
    strides = @pyv `$(info.dict).get("strides")`::Union{Nothing,NTuple{N,Int}}
    if strides === nothing
        size = pyarray_size(info, Val(N))
        elsize = pyarray_elsize(info)
        size_to_cstrides(elsize, size...)
    else
        strides
    end
end
pyarray_handle(info::PyArrayInfo_ArrayInterface) = info.handle
pyarray_elsize(info::PyArrayInfo_ArrayInterface) = begin
    sz = @pya ```
    s = $(info.dict)["typestr"]
    c = s[1]
    if c == "O":
        ans = -1
    elif c == "m" or c == "M":
        ans = int(s[2:].split("[")[0])
    else:
        ans = int(s[2:])
    ```::Int
    sz < 0 ? sizeof(CPyPtr) : sz
end

pyarray_info(ref; buffer::Bool=true, array::Bool=true, copy::Bool=true) = begin
    # __array_interface__
    if array
        @py ```
        try:
            d = $ref.__array_interface__
            assert d["version"] == 3
            assert d.get("mask") is None
        except:
            d = None
        $(d::Union{Nothing,PyObject}) = d
        ```
        if d !== nothing
            @py ```
            data = $d.get("data", $ref)
            if isinstance(data, tuple):
                ptr = data[0]
                mut = not data[1]
                buf = None
            else:
                ptr = 0
                mut = False
                buf = data
            $(ptr::UInt) = ptr
            $(mut::Bool) = mut
            $(buf::Union{PyRef,Nothing}) = buf
            ```
            if buf === nothing
                return PyArrayInfo_ArrayInterface(d, Ptr{Cvoid}(ptr), mut, (ref, d))
            else
                b = PyBuffer(buf, C.PyBUF_RECORDS_RO)
                offset = @pyv `$d.get("offset", 0)`::Int
                return PyArrayInfo_ArrayInterface(d, b.buf+offset, !b.readonly, (ref, d, b))
            end
        end
    end
    # __array_struct__
    if array
        @py ```
        d = None
        try:
            d = $ref.__array_struct__
        except:
            pass
        $(d::Union{Nothing,PyObject}) = d
        ```
        d === nothing || error("not implemented: creating PyArray from __array_struct__")
    end
    # buffer protocol
    if buffer
        b = PyBuffer()
        err = C.PyObject_GetBuffer(ref, pointer(b.info), C.PyBUF_RECORDS_RO)
        if ism1(err)
            C.PyErr_Clear()
        else
            return PyArrayInfo_Buffer(b)
        end
    end
    # __array__
    if array && copy
        @py ```
        a = None
        try:
            a = $ref.__array__()
        except:
            pass
        $(a::Union{Nothing,PyRef}) = a
        ```
        a === nothing || return pyarray_info(a, buffer=buffer, array=array, copy=false)
    end
    error("this object cannot be interpreted as a strided array")
end

Base.size(x::PyArray) = x.size
Base.length(x::PyArray) = x.length
ismutablearray(::PyArray{R,N,T,M,L}) where {R,N,T,M,L} = M
Base.IndexStyle(::Type{PyArray{R,N,T,M,L}}) where {R,N,T,M,L} =
    L ? Base.IndexLinear() : Base.IndexCartesian()

Base.@propagate_inbounds Base.getindex(
    x::PyArray{R,N,T,M,L},
    i::Vararg{Int,N},
) where {R,N,T,M,L} = pyarray_getindex(x, i...)
Base.@propagate_inbounds Base.getindex(
    x::PyArray{R,N,T,M,true},
    i::Int
) where {R,N,T,M} = pyarray_getindex(x, i)
Base.@propagate_inbounds Base.getindex(
    x::PyArray{R,1,T,M,true},
    i::Int
) where {R,T,M} = pyarray_getindex(x, i)

Base.@propagate_inbounds pyarray_getindex(x::PyArray, i...) = begin
    @boundscheck checkbounds(x, i...)
    pyarray_load(eltype(x), x.buf + pyarray_offset(x, i...))
end

Base.@propagate_inbounds Base.setindex!(
    x::PyArray{R,N,T,true,L},
    v,
    i::Vararg{Int,N},
) where {R,N,T,L} = pyarray_setindex!(x, v, i...)
Base.@propagate_inbounds Base.setindex!(
    x::PyArray{R,N,T,true,true},
    v,
    i::Int
) where {R,N,T} = pyarray_setindex!(x, v, i)
Base.@propagate_inbounds Base.setindex!(
    x::PyArray{R,1,T,true,true},
    v,
    i::Int
) where {R,T} = pyarray_setindex!(x, v, i)

Base.@propagate_inbounds pyarray_setindex!(x::PyArray, v, i...) = begin
    @boundscheck checkbounds(x, i...)
    pyarray_store!(x.buf + pyarray_offset(x, i...), convert(eltype(x), v))
    x
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

pyarray_offset(x::PyArray{R,N,T,M,true}, i::Int) where {R,N,T,M} =
    N == 0 ? 0 : (i - 1) * x.bytestrides[1]

pyarray_offset(x::PyArray{R,1,T,M,true}, i::Int) where {R,T,M} = (i - 1) .* x.bytestrides[1]

pyarray_offset(x::PyArray{R,N}, i::Vararg{Int,N}) where {R,N} =
    sum((i .- 1) .* x.bytestrides)
