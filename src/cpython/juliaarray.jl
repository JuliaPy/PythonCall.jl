const PyJuliaArrayValue_Type__ref = Ref(PyNULL)
PyJuliaArrayValue_Type() = begin
    ptr = PyJuliaArrayValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyNULL
        t = fill(
            PyType_Create(
                c,
                name = "julia.ArrayValue",
                base = base,
                as_mapping = (
                    subscript = pyjlarray_getitem,
                    ass_subscript = pyjlarray_setitem,
                ),
                as_buffer = (
                    get = pyjlarray_get_buffer,
                    release = pyjlarray_release_buffer,
                ),
                getset = [
                    (name = "ndim", get = pyjlarray_ndim),
                    (name = "shape", get = pyjlarray_shape),
                    (name = "__array_interface__", get = pyjlarray_array_interface),
                ],
                methods = [
                    (name = "copy", flags = Py_METH_NOARGS, meth = pyjlarray_copy),
                    (name = "reshape", flags = Py_METH_O, meth = pyjlarray_reshape),
                    (name = "__array__", flags = Py_METH_NOARGS, meth = pyjlarray_array),
                ],
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyNULL
        abc = PyCollectionABC_Type()
        isnull(abc) && return PyNULL
        ism1(PyABC_Register(ptr, abc)) && return PyNULL
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaArrayValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaArrayValue_New(x::AbstractArray) = PyJuliaValue_New(PyJuliaArrayValue_Type(), x)
PyJuliaValue_From(x::AbstractArray) = PyJuliaArrayValue_New(x)

pyjl_getaxisindex(x::AbstractUnitRange{<:Integer}, ko::PyPtr) = begin
    if PySlice_Check(ko)
        ao, co, bo = PySimpleObject_GetValue(ko, Tuple{PyPtr,PyPtr,PyPtr})
        # start
        r = PyObject_TryConvert(ao, Union{Int,Nothing})
        r == -1 && return PYERR()
        r == 0 && (
            PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR()
        )
        a = takeresult(Union{Int,Nothing})
        # step
        r = PyObject_TryConvert(bo, Union{Int,Nothing})
        r == -1 && return PYERR()
        r == 0 && (
            PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR()
        )
        b = takeresult(Union{Int,Nothing})
        # stop
        r = PyObject_TryConvert(co, Union{Int,Nothing})
        r == -1 && return PYERR()
        r == 0 && (
            PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR()
        )
        c = takeresult(Union{Int,Nothing})
        # step defaults to 1
        b′ = b === nothing ? 1 : b
        if a === nothing && c === nothing
            # when neither is specified, start and stop default to the full range,
            # which is reversed when the step is negative
            if b′ > 0
                a′ = Int(first(x))
                c′ = Int(last(x))
            elseif b′ < 0
                a′ = Int(last(x))
                c′ = Int(first(x))
            else
                PyErr_SetString(PyExc_ValueError(), "step must be non-zero")
                return PYERR()
            end
        else
            # start defaults
            a′ = Int(a === nothing ? first(x) : a < 0 ? (last(x) + a + 1) : (first(x) + a))
            c′ = Int(
                c === nothing ? last(x) :
                c < 0 ? (last(x) + 1 + c - sign(b′)) : (first(x) + c - sign(b′)),
            )
        end
        r = a′:b′:c′
        if !checkbounds(Bool, x, r)
            PyErr_SetString(PyExc_IndexError(), "array index out of bounds")
            return PYERR()
        end
        r
    else
        r = PyObject_TryConvert(ko, Int)
        ism1(r) && return PYERR()
        r == 0 && (
            PyErr_SetString(
                PyExc_TypeError(),
                "index must be slice or integer, got $(PyType_Name(Py_Type(ko)))",
            );
            return PYERR()
        )
        k = takeresult(Int)
        k′ = k < 0 ? (last(x) + k + 1) : (first(x) + k)
        checkbounds(Bool, x, k′) || (
            PyErr_SetString(PyExc_IndexError(), "array index out of bounds"); return PYERR()
        )
        k′
    end
end

pyjl_getarrayindices(x::AbstractArray, ko::PyPtr) = begin
    kos = PyTuple_Check(ko) ? [PyTuple_GetItem(ko, i - 1) for i = 1:PyTuple_Size(ko)] : [ko]
    length(kos) == ndims(x) || (
        PyErr_SetString(
            PyExc_TypeError(),
            "expecting exactly $(ndims(x)) indices, got $(length(kos))",
        );
        return PYERR()
    )
    ks = []
    for (i, ko) in enumerate(kos)
        k = pyjl_getaxisindex(axes(x, i), ko)
        k === PYERR() && return PYERR()
        push!(ks, k)
    end
    ks
end

pyjlarray_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    k = pyjl_getarrayindices(x, ko)
    k === PYERR() && return PyNULL
    try
        if all(x -> x isa Int, k)
            PyObject_From(x[k...])
        else
            PyObject_From(view(x, k...))
        end
    catch err
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end
end

pyjlarray_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    k = pyjl_getarrayindices(x, ko)
    k === PYERR() && return Cint(-1)
    try
        if isnull(vo)
            deleteat!(x, k...)
            Cint(0)
        else
            ism1(PyObject_Convert(vo, eltype(x))) && return Cint(-1)
            v = takeresult(eltype(x))
            if all(x -> x isa Int, k)
                x[k...] = v
                Cint(0)
            else
                PyErr_SetString(PyExc_TypeError(), "multiple assignment not supported")
                Cint(-1)
            end
        end
    catch err
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        elseif err isa MethodError && err.f === deleteat!
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Cint(-1)
    end
end

pyjlarray_ndim(xo::PyPtr, ::Ptr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    PyObject_From(ndims(x))
end

pyjlarray_shape(xo::PyPtr, ::Ptr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    PyObject_From(size(x))
end

pyjlarray_copy(xo::PyPtr, ::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractArray
        PyObject_From(copy(x))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlarray_reshape(xo::PyPtr, arg::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractArray
        r = PyObject_TryConvert(arg, Union{Int,Tuple{Vararg{Int}}})
        r == -1 && return PyNULL
        r == 0 && (
            PyErr_SetString(
                PyExc_TypeError(),
                "shape must be an integer or tuple of integers",
            );
            return PyNULL
        )
        PyObject_From(reshape(x, takeresult()...))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

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
        if Python.size_to_cstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not C contiguous and strides not requested")
            return Cint(-1)
        end
        b.strides[] = C_NULL
    end

    # check contiguity
    if isflagset(flags, PyBUF_C_CONTIGUOUS)
        if Python.size_to_cstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not C contiguous")
            return Cint(-1)
        end
    end
    if isflagset(flags, PyBUF_F_CONTIGUOUS)
        if Python.size_to_fstrides(elsz, sz...) != strds
            PyErr_SetString(PyExc_BufferError(), "not Fortran contiguous")
            return Cint(-1)
        end
    end
    if isflagset(flags, PyBUF_ANY_CONTIGUOUS)
        if Python.size_to_cstrides(elsz, sz...) != strds &&
           Python.size_to_fstrides(elsz, sz...) != strds
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
    Python.allocatedinline(T) &&
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
                Python.pybufferformat(eltype(x)),
                size(x),
                strides(x) .* Python.aligned_sizeof(eltype(x)),
                Python.ismutablearray(x),
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

pyjlarray_isarrayabletype(::Type{T}) where {T} = T in (
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Bool,
    Float16,
    Float32,
    Float64,
    Complex{Float16},
    Complex{Float32},
    Complex{Float64},
)
pyjlarray_isarrayabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) &&
    Python.allocatedinline(T) &&
    all(pyjlarray_isarrayabletype, T.parameters)
pyjlarray_isarrayabletype(::Type{NamedTuple{names,types}}) where {names,types} =
    pyjlarray_isarrayabletype(types)

pyjlarray_array_interface(xo::PyPtr, ::Ptr{Cvoid}) =
    _pyjlarray_array_interface(PyJuliaValue_GetValue(xo)::AbstractArray)

PyDescrObject_From(fields) = begin
    ro = PyList_New(0)
    for (name, descr) in fields
        descro = descr isa String ? PyUnicode_From(descr) : PyDescrObject_From(descr)
        isnull(descro) && (Py_DecRef(ro); return PyNULL)
        fieldo = PyTuple_From((name, PyObjectRef(descro)))
        Py_DecRef(descro)
        isnull(fieldo) && (Py_DecRef(ro); return PyNULL)
        err = PyList_Append(ro, fieldo)
        ism1(err) && (Py_DecRef(ro); return PyNULL)
    end
    ro
end

_pyjlarray_array_interface(x::AbstractArray) =
    try
        if pyjlarray_isarrayabletype(eltype(x))
            # gather information
            shape = size(x)
            data = (UInt(Base.unsafe_convert(Ptr{eltype(x)}, x)), !Python.ismutablearray(x))
            strides = Base.strides(x) .* Python.aligned_sizeof(eltype(x))
            version = 3
            typestr, descr = Python.pytypestrdescr(eltype(x))
            isempty(typestr) && error("invalid element type")
            # make the dictionary
            d = PyDict_From(
                Dict(
                    "shape" => shape,
                    "typestr" => typestr,
                    "data" => data,
                    "strides" => strides,
                    "version" => version,
                ),
            )
            isnull(d) && return PyNULL
            if descr !== nothing
                descro = PyDescrObject_From(descr)
                err = PyDict_SetItemString(d, "descr", descro)
                Py_DecRef(descro)
                ism1(err) && (Py_DecRef(d); return PyNULL)
            end
            d
        else
            error("invalid element type")
        end
    catch err
        PyErr_SetString(PyExc_AttributeError(), "__array_interface__")
        PyNULL
    end

pyjlarray_array(xo::PyPtr, ::PyPtr) = begin
    if PyObject_HasAttrString(xo, "__array_interface__") == 0
        # convert to a PyObjectArray
        x = PyJuliaValue_GetValue(xo)::AbstractArray
        y = try
            Python.PyObjectArray(x)
        catch err
            PyErr_SetJuliaError(err)
            return PyNULL
        end
        yo = PyJuliaArrayValue_New(y)
        isnull(yo) && return PyNULL
    else
        # already supports the array interface
        Py_IncRef(xo)
        yo = xo
    end
    npo = PyImport_ImportModule("numpy")
    isnull(npo) && (Py_DecRef(yo); return PyNULL)
    aao = PyObject_GetAttrString(npo, "array")
    Py_DecRef(npo)
    isnull(aao) && (Py_DecRef(yo); return PyNULL)
    ao = PyObject_CallNice(aao, PyObjectRef(yo))
    Py_DecRef(aao)
    Py_DecRef(yo)
    ao
end
