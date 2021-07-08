### Buffer Protocol

isflagset(flags, mask) = (flags & mask) == mask

const PYJLBUFCACHE = Dict{Ptr{Cvoid},Any}()

pyjl_get_buffer_impl(
    o::PyPtr,
    buf::Ptr{Py_buffer},
    flags,
    ptr,
    elsz,
    len,
    ndim,
    fmt,
    sz,
    strds,
    mutable,
) = begin
    b = UnsafePtr(buf)
    c = []

    # not influenced by flags: obj, buf, len, itemsize, ndim
    b.obj[] = C_NULL
    b.buf[] = ptr
    b.itemsize[] = elsz
    b.len[] = elsz * len
    b.ndim[] = ndim

    # readonly
    if isflagset(flags, PyBUF_WRITABLE)
        if mutable
            b.readonly[] = 1
        else
            PyErr_SetString(PyExc_BufferError(), "not writable")
            return Cint(-1)
        end
    else
        b.readonly[] = mutable ? 0 : 1
    end

    # format
    if isflagset(flags, PyBUF_FORMAT)
        b.format[] = cacheptr!(c, fmt)
    else
        b.format[] = C_NULL
    end

    # shape
    if isflagset(flags, PyBUF_ND)
        b.shape[] = cacheptr!(c, Py_ssize_t[sz...])
    else
        b.shape[] = C_NULL
    end

    # strides
    if isflagset(flags, PyBUF_STRIDES)
        b.strides[] = cacheptr!(c, Py_ssize_t[strds...])
    else
        if PythonCall.size_to_cstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not C contiguous and strides not requested")
            return Cint(-1)
        end
        b.strides[] = C_NULL
    end

    # check contiguity
    if isflagset(flags, PyBUF_C_CONTIGUOUS)
        if PythonCall.size_to_cstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not C contiguous")
            return Cint(-1)
        end
    end
    if isflagset(flags, PyBUF_F_CONTIGUOUS)
        if PythonCall.size_to_fstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not Fortran contiguous")
            return Cint(-1)
        end
    end
    if isflagset(flags, PyBUF_ANY_CONTIGUOUS)
        if PythonCall.size_to_cstrides(elsz, sz...) != strds &&
           PythonCall.size_to_fstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not contiguous")
            return Cint(-1)
        end
    end

    # suboffsets
    b.suboffsets[] = C_NULL

    # internal
    cptr = Base.pointer_from_objref(c)
    PYJLBUFCACHE[cptr] = c
    b.internal[] = cptr

    # obj
    Py_IncRef(o)
    b.obj[] = o
    Cint(0)
end

pyjlarray_isbufferabletype(::Type{T}) where {T} = T in (
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex{Float16},
    Complex{Float32},
    Complex{Float64},
    Bool,
    Ptr{Cvoid},
)
pyjlarray_isbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) &&
    PythonCall.allocatedinline(T) &&
    all(pyjlarray_isbufferabletype, fieldtypes(T))
pyjlarray_isbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjlarray_isbufferabletype(T)

_pyjlarray_get_buffer(xo, buf, flags, x::AbstractArray) =
    try
        if pyjlarray_isbufferabletype(eltype(x))
            pyjl_get_buffer_impl(
                xo,
                buf,
                flags,
                Base.unsafe_convert(Ptr{eltype(x)}, x),
                sizeof(eltype(x)),
                length(x),
                ndims(x),
                PythonCall.pybufferformat(eltype(x)),
                size(x),
                strides(x) .* PythonCall.aligned_sizeof(eltype(x)),
                PythonCall.ismutablearray(x),
            )
        else
            error("element type is not bufferable")
        end
    catch err
        PyErr_SetString(
            PyExc_BufferError(),
            "Buffer protocol not supported by Julia '$(typeof(x))' (details: $err)",
        )
        Cint(-1)
    end

pyjlarray_get_buffer(xo::PyPtr, buf::Ptr{Py_buffer}, flags::Cint) =
    _pyjlarray_get_buffer(xo, buf, flags, PyJuliaValue_GetValue(xo)::AbstractArray)

pyjlarray_release_buffer(xo::PyPtr, buf::Ptr{Py_buffer}) = begin
    delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[!])
    nothing
end

### Array Interface
