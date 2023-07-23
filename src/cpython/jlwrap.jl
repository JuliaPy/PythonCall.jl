# we store the actual julia values here
# the `value` field of `PyJuliaValueObject` indexes into here
const PYJLVALUES = []
# unused indices in PYJLVALUES
const PYJLFREEVALUES = Int[]

function _pyjl_new(t::PyPtr, ::PyPtr, ::PyPtr)
    o = ccall(UnsafePtr{PyTypeObject}(t).alloc[!], PyPtr, (PyPtr, Py_ssize_t), t, 0)
    o == PyNULL && return PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = PyNULL
    UnsafePtr{PyJuliaValueObject}(o).value[] = 0
    return o
end

function _pyjl_dealloc(o::PyPtr)
    idx = UnsafePtr{PyJuliaValueObject}(o).value[]
    if idx != 0
        PYJLVALUES[idx] = nothing
        push!(PYJLFREEVALUES, idx)
    end
    UnsafePtr{PyJuliaValueObject}(o).weaklist[!] == PyNULL || PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{PyTypeObject}(Py_Type(o)).free[!], Cvoid, (PyPtr,), o)
    nothing
end

const PYJLMETHODS = Vector{Any}()

function PyJulia_MethodNum(f)
    @nospecialize f
    push!(PYJLMETHODS, f)
    return length(PYJLMETHODS)
end

function _pyjl_isnull(o::PyPtr, ::PyPtr)
    ans = PyJuliaValue_IsNull(o) ? POINTERS._Py_TrueStruct : POINTERS._Py_FalseStruct
    Py_IncRef(ans)
    ans
end

function _pyjl_callmethod(o::PyPtr, args::PyPtr)
    nargs = PyTuple_Size(args)
    @assert nargs > 0
    num = PyLong_AsLongLong(PyTuple_GetItem(args, 0))
    num == -1 && return PyNULL
    f = PYJLMETHODS[num]
    # this form gets defined in jlwrap/base.jl
    return _pyjl_callmethod(f, o, args, nargs)::PyPtr
end

const PYJLBUFCACHE = Dict{Ptr{Cvoid},Any}()

@kwdef struct PyBufferInfo{N}
    # data
    ptr::Ptr{Cvoid}
    readonly::Bool
    # items
    itemsize::Int
    format::String
    # layout
    shape::NTuple{N,Int}
    strides::NTuple{N,Int}
    suboffsets::NTuple{N,Int} = ntuple(i -> -1, N)
end

_pyjl_get_buffer_impl(obj::PyPtr, buf::Ptr{Py_buffer}, flags::Cint, x, f) = _pyjl_get_buffer_impl(obj, buf, flags, f(x)::PyBufferInfo)

function _pyjl_get_buffer_impl(obj::PyPtr, buf::Ptr{Py_buffer}, flags::Cint, info::PyBufferInfo{N}) where {N}
    b = UnsafePtr(buf)
    c = []

    # not influenced by flags: obj, buf, len, itemsize, ndim
    b.obj[] = C_NULL
    b.buf[] = info.ptr
    b.itemsize[] = info.itemsize
    b.len[] = info.itemsize * prod(info.shape)
    b.ndim[] = N

    # readonly
    if !info.readonly
        b.readonly[] = 0
    elseif Utils.isflagset(flags, PyBUF_WRITABLE)
        PyErr_SetString(POINTERS.PyExc_BufferError, "not writable")
        return Cint(-1)
    else
        b.readonly[] = 1
    end

    # format
    if Utils.isflagset(flags, PyBUF_FORMAT)
        push!(c, info.format)
        b.format[] = pointer(info.format)
    else
        b.format[] = C_NULL
    end

    # shape
    if Utils.isflagset(flags, PyBUF_ND)
        shape = Py_ssize_t[info.shape...]
        push!(c, shape)
        b.shape[] = pointer(shape)
    else
        b.shape[] = C_NULL
    end

    # strides
    if Utils.isflagset(flags, PyBUF_STRIDES)
        strides = Py_ssize_t[info.strides...]
        push!(c, strides)
        b.strides[] = pointer(strides)
    elseif Utils.size_to_cstrides(info.itemsize, info.shape) == info.strides
        b.strides[] = C_NULL
    else
        PyErr_SetString(POINTERS.PyExc_BufferError, "not C contiguous and strides not requested")
        return Cint(-1)
    end

    # suboffsets
    if all(==(-1), info.suboffsets)
        b.suboffsets[] = C_NULL
    elseif Utils.isflagset(flags, PyBUF_INDIRECT)
        suboffsets = Py_ssize_t[info.suboffsets...]
        push!(c, suboffsets)
        b.suboffsets[] = pointer(suboffsets)
    else
        PyErr_SetString(POINTERS.PyExc_BufferError, "indirect array and suboffsets not requested")
        return Cint(-1)
    end

    # check contiguity
    if Utils.isflagset(flags, PyBUF_C_CONTIGUOUS)
        if Utils.size_to_cstrides(info.itemsize, info.shape) != info.strides
            PyErr_SetString(POINTERS.PyExc_BufferError, "not C contiguous")
            return Cint(-1)
        end
    end
    if Utils.isflagset(flags, PyBUF_F_CONTIGUOUS)
        if Utils.size_to_fstrides(info.itemsize, info.shape) != info.strides
            PyErr_SetString(POINTERS.PyExc_BufferError, "not Fortran contiguous")
            return Cint(-1)
        end
    end
    if Utils.isflagset(flags, PyBUF_ANY_CONTIGUOUS)
        if Utils.size_to_cstrides(info.itemsize, info.shape) != info.strides &&
           Utils.size_to_fstrides(info.itemsize, info.shape) != info.strides
            PyErr_SetString(POINTERS.PyExc_BufferError, "not contiguous")
            return Cint(-1)
        end
    end

    # internal
    cptr = Base.pointer_from_objref(c)
    PYJLBUFCACHE[cptr] = c
    b.internal[] = cptr

    # obj
    Py_IncRef(obj)
    b.obj[] = obj
    Cint(0)
end

function _pyjl_get_buffer(o::PyPtr, buf::Ptr{Py_buffer}, flags::Cint)
    num_ = PyObject_GetAttrString(o, "_jl_buffer_info")
    num_ == C_NULL && (PyErr_Clear(); PyErr_SetString(POINTERS.PyExc_BufferError, "not a buffer"); return Cint(-1))
    num = PyLong_AsLongLong(num_)
    Py_DecRef(num_)
    num == -1 && return Cint(-1)
    try
        f = PYJLMETHODS[num]
        x = PyJuliaValue_GetValue(o)
        return _pyjl_get_buffer_impl(o, buf, flags, x, f)::Cint
    catch exc
        @debug "error getting the buffer" exc
        PyErr_SetString(POINTERS.PyExc_BufferError, "some error occurred getting the buffer")
        return Cint(-1)
    end
end

function _pyjl_release_buffer(xo::PyPtr, buf::Ptr{Py_buffer})
    delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[!])
    nothing
end

function _pyjl_reduce(self::PyPtr, ::PyPtr)
    v = _pyjl_serialize(self, PyNULL)
    v == PyNULL && return PyNULL
    args = PyTuple_New(1)
    args == PyNULL && (Py_DecRef(v); return PyNULL)
    err = PyTuple_SetItem(args, 0, v)
    err == -1 && (Py_DecRef(args); return PyNULL)
    red = PyTuple_New(2)
    red == PyNULL && (Py_DecRef(args); return PyNULL)
    err = PyTuple_SetItem(red, 1, args)
    err == -1 && (Py_DecRef(red); return PyNULL)
    f = PyObject_GetAttrString(self, "_jl_deserialize")
    f == PyNULL && (Py_DecRef(red); return PyNULL)
    err = PyTuple_SetItem(red, 0, f)
    err == -1 && (Py_DecRef(red); return PyNULL)
    return red
end

function _pyjl_serialize(self::PyPtr, ::PyPtr)
    try
        io = IOBuffer()
        serialize(io, PyJuliaValue_GetValue(self))
        b = take!(io)
        return PyBytes_FromStringAndSize(pointer(b), sizeof(b))
    catch e
        PyErr_SetString(POINTERS.PyExc_Exception, "error serializing this value")
        # wrap sprint in another try-catch block to prevent this function from throwing
        try
            @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
        catch e2
            @debug "Error printing exception: $e2"
        end
        return PyNULL
    end
end

function _pyjl_deserialize(t::PyPtr, v::PyPtr)
    try
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyBytes_AsStringAndSize(v, ptr, len)
        err == -1 && return PyNULL
        io = IOBuffer(unsafe_wrap(Array, Ptr{UInt8}(ptr[]), Int(len[])))
        x = deserialize(io)
        return PyJuliaValue_New(t, x)
    catch e
        PyErr_SetString(POINTERS.PyExc_Exception, "error deserializing this value")
        # wrap sprint in another try-catch block to prevent this function from throwing
        try
            @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
        catch e2
            @debug "Error printing exception: $e2"
        end
        return PyNULL
    end
end

const _pyjlbase_name = "juliacall.ValueBase"
const _pyjlbase_type = fill(C.PyTypeObject())
const _pyjlbase_isnull_name = "_jl_isnull"
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_reduce_name = "__reduce__"
const _pyjlbase_serialize_name = "_jl_serialize"
const _pyjlbase_deserialize_name = "_jl_deserialize"
const _pyjlbase_methods = Vector{PyMethodDef}()
const _pyjlbase_as_buffer = fill(PyBufferProcs())

function init_jlwrap()
    empty!(_pyjlbase_methods)
    push!(_pyjlbase_methods,
        PyMethodDef(
            name = pointer(_pyjlbase_callmethod_name),
            meth = @cfunction(_pyjl_callmethod, PyPtr, (PyPtr, PyPtr)),
            flags = Py_METH_VARARGS,
        ),
        PyMethodDef(
            name = pointer(_pyjlbase_isnull_name),
            meth = @cfunction(_pyjl_isnull, PyPtr, (PyPtr, PyPtr)),
            flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            name = pointer(_pyjlbase_reduce_name),
            meth = @cfunction(_pyjl_reduce, PyPtr, (PyPtr, PyPtr)),
            flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            name = pointer(_pyjlbase_serialize_name),
            meth = @cfunction(_pyjl_serialize, PyPtr, (PyPtr, PyPtr)),
            flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            name = pointer(_pyjlbase_deserialize_name),
            meth = @cfunction(_pyjl_deserialize, PyPtr, (PyPtr, PyPtr)),
            flags = Py_METH_O | Py_METH_CLASS,
        ),
        PyMethodDef(),
    )
    _pyjlbase_as_buffer[] = PyBufferProcs(
        get = @cfunction(_pyjl_get_buffer, Cint, (PyPtr, Ptr{Py_buffer}, Cint)),
        release = @cfunction(_pyjl_release_buffer, Cvoid, (PyPtr, Ptr{Py_buffer})),
    )
    _pyjlbase_type[] = PyTypeObject(
        name = pointer(_pyjlbase_name),
        basicsize = sizeof(PyJuliaValueObject),
        # new = POINTERS.PyType_GenericNew,
        new = @cfunction(_pyjl_new, PyPtr, (PyPtr, PyPtr, PyPtr)),
        dealloc = @cfunction(_pyjl_dealloc, Cvoid, (PyPtr,)),
        flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_VERSION_TAG,
        weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
        # getattro = POINTERS.PyObject_GenericGetAttr,
        # setattro = POINTERS.PyObject_GenericSetAttr,
        methods = pointer(_pyjlbase_methods),
        as_buffer = pointer(_pyjlbase_as_buffer),
    )
    o = POINTERS.PyJuliaBase_Type = PyPtr(pointer(_pyjlbase_type))
    if PyType_Ready(o) == -1
        PyErr_Print()
        error("Error initializing 'juliacall.ValueBase'")
    end
end

PyJuliaValue_IsNull(o::PyPtr) = UnsafePtr{PyJuliaValueObject}(o).value[] == 0

PyJuliaValue_GetValue(o::PyPtr) = PYJLVALUES[UnsafePtr{PyJuliaValueObject}(o).value[]]

PyJuliaValue_SetValue(o::PyPtr, @nospecialize(v)) = begin
    idx = UnsafePtr{PyJuliaValueObject}(o).value[]
    if idx == 0
        if isempty(PYJLFREEVALUES)
            push!(PYJLVALUES, v)
            idx = length(PYJLVALUES)
        else
            idx = pop!(PYJLFREEVALUES)
            PYJLVALUES[idx] = v
        end
        UnsafePtr{PyJuliaValueObject}(o).value[] = idx
    else
        PYJLVALUES[idx] = v
    end
    nothing
end

PyJuliaValue_New(t::PyPtr, @nospecialize(v)) = begin
    if PyType_IsSubtype(t, POINTERS.PyJuliaBase_Type) != 1
        PyErr_SetString(POINTERS.PyExc_TypeError, "Expecting a subtype of 'juliacall.ValueBase'")
        return PyNULL
    end
    o = PyObject_CallObject(t, PyNULL)
    o == PyNULL && return PyNULL
    PyJuliaValue_SetValue(o, v)
    return o
end
