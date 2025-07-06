module Cjl

using ...C: C
using ...Utils: Utils
using Base: @kwdef
using UnsafePointers: UnsafePtr
using Serialization: serialize, deserialize

@kwdef struct PyJuliaValueObject
    ob_base::C.PyObject = C.PyObject()
    value::Int = 0
    weaklist::C.PyPtr = C_NULL
end

const PyJuliaBase_Type = Ref(C.PyNULL)

# we store the actual julia values here
# the `value` field of `PyJuliaValueObject` indexes into here
const PYJLVALUES = IdDict{Int,Any}()
# unused indices in PYJLVALUES
const PYJLFREEVALUES = Int[]
# Thread safety for PYJLVALUES and PYJLFREEVALUES
const PYJLVALUES_LOCK = Threads.SpinLock()
# Track next available index
const PYJLVALUES_NEXT_IDX = Ref(1)

function _pyjl_new(t::C.PyPtr, ::C.PyPtr, ::C.PyPtr)
    o = ccall(UnsafePtr{C.PyTypeObject}(t).alloc[!], C.PyPtr, (C.PyPtr, C.Py_ssize_t), t, 0)
    o == C.PyNULL && return C.PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C.PyNULL
    UnsafePtr{PyJuliaValueObject}(o).value[] = 0
    return o
end

function _pyjl_dealloc(o::C.PyPtr)
    idx = UnsafePtr{PyJuliaValueObject}(o).value[]
    if idx != 0
        Base.@lock PYJLVALUES_LOCK begin
            delete!(PYJLVALUES, idx)
            push!(PYJLFREEVALUES, idx)
        end
    end
    UnsafePtr{PyJuliaValueObject}(o).weaklist[!] == C.PyNULL || C.PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{C.PyTypeObject}(C.Py_Type(o)).free[!], Cvoid, (C.PyPtr,), o)
    nothing
end

const PYJLMETHODS = Vector{Any}()
const PYJLMETHODS_LOCK = Threads.SpinLock()

function PyJulia_MethodNum(f)
    @nospecialize f
    Base.@lock PYJLMETHODS_LOCK begin
        push!(PYJLMETHODS, f)
        return length(PYJLMETHODS)
    end
end

function _pyjl_isnull(o::C.PyPtr, ::C.PyPtr)
    ans = PyJuliaValue_IsNull(o) ? C.POINTERS._Py_TrueStruct : C.POINTERS._Py_FalseStruct
    C.Py_IncRef(ans)
    ans
end

function _pyjl_callmethod(o::C.PyPtr, args::C.PyPtr)
    nargs = C.PyTuple_Size(args)
    @assert nargs > 0
    num = C.PyLong_AsLongLong(C.PyTuple_GetItem(args, 0))
    num == -1 && return C.PyNULL
    f = Base.@lock PYJLMETHODS_LOCK PYJLMETHODS[num]
    # this form gets defined in jlwrap/base.jl
    return _pyjl_callmethod(f, o, args, nargs)::C.PyPtr
end

const PYJLBUFCACHE = Dict{Ptr{Cvoid},Any}()
const PYJLBUFCACHE_LOCK = Threads.SpinLock()

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

_pyjl_get_buffer_impl(obj::C.PyPtr, buf::Ptr{C.Py_buffer}, flags::Cint, x, f) =
    _pyjl_get_buffer_impl(obj, buf, flags, f(x)::PyBufferInfo)

function _pyjl_get_buffer_impl(
    obj::C.PyPtr,
    buf::Ptr{C.Py_buffer},
    flags::Cint,
    info::PyBufferInfo{N},
) where {N}
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
    elseif Utils.isflagset(flags, C.PyBUF_WRITABLE)
        C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not writable")
        return Cint(-1)
    else
        b.readonly[] = 1
    end

    # format
    if Utils.isflagset(flags, C.PyBUF_FORMAT)
        push!(c, info.format)
        b.format[] = pointer(info.format)
    else
        b.format[] = C_NULL
    end

    # shape
    if Utils.isflagset(flags, C.PyBUF_ND)
        shape = C.Py_ssize_t[info.shape...]
        push!(c, shape)
        b.shape[] = pointer(shape)
    else
        b.shape[] = C_NULL
    end

    # strides
    if Utils.isflagset(flags, C.PyBUF_STRIDES)
        strides = C.Py_ssize_t[info.strides...]
        push!(c, strides)
        b.strides[] = pointer(strides)
    elseif Utils.size_to_cstrides(info.itemsize, info.shape) == info.strides
        b.strides[] = C_NULL
    else
        C.PyErr_SetString(
            C.POINTERS.PyExc_BufferError,
            "not C contiguous and strides not requested",
        )
        return Cint(-1)
    end

    # suboffsets
    if all(==(-1), info.suboffsets)
        b.suboffsets[] = C_NULL
    elseif Utils.isflagset(flags, C.PyBUF_INDIRECT)
        suboffsets = C.Py_ssize_t[info.suboffsets...]
        push!(c, suboffsets)
        b.suboffsets[] = pointer(suboffsets)
    else
        C.PyErr_SetString(
            C.POINTERS.PyExc_BufferError,
            "indirect array and suboffsets not requested",
        )
        return Cint(-1)
    end

    # check contiguity
    if Utils.isflagset(flags, C.PyBUF_C_CONTIGUOUS)
        if Utils.size_to_cstrides(info.itemsize, info.shape) != info.strides
            C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not C contiguous")
            return Cint(-1)
        end
    end
    if Utils.isflagset(flags, C.PyBUF_F_CONTIGUOUS)
        if Utils.size_to_fstrides(info.itemsize, info.shape) != info.strides
            C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not Fortran contiguous")
            return Cint(-1)
        end
    end
    if Utils.isflagset(flags, C.PyBUF_ANY_CONTIGUOUS)
        if Utils.size_to_cstrides(info.itemsize, info.shape) != info.strides &&
           Utils.size_to_fstrides(info.itemsize, info.shape) != info.strides
            C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not contiguous")
            return Cint(-1)
        end
    end

    # internal
    cptr = Base.pointer_from_objref(c)
    Base.@lock PYJLBUFCACHE_LOCK begin
        PYJLBUFCACHE[cptr] = c
    end
    b.internal[] = cptr

    # obj
    C.Py_IncRef(obj)
    b.obj[] = obj
    Cint(0)
end

function _pyjl_get_buffer(o::C.PyPtr, buf::Ptr{C.Py_buffer}, flags::Cint)
    num_ = C.PyObject_GetAttrString(o, "_jl_buffer_info")
    num_ == C_NULL && (
        C.PyErr_Clear(); C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not a buffer"); return Cint(-1)
    )
    num = C.PyLong_AsLongLong(num_)
    C.Py_DecRef(num_)
    num == -1 && return Cint(-1)
    try
        f = Base.@lock PYJLMETHODS_LOCK PYJLMETHODS[num]
        x = PyJuliaValue_GetValue(o)
        return _pyjl_get_buffer_impl(o, buf, flags, x, f)::Cint
    catch exc
        @debug "error getting the buffer" exc
        C.PyErr_SetString(
            C.POINTERS.PyExc_BufferError,
            "some error occurred getting the buffer",
        )
        return Cint(-1)
    end
end

function _pyjl_release_buffer(xo::C.PyPtr, buf::Ptr{C.Py_buffer})
    Base.@lock PYJLBUFCACHE_LOCK begin
        delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[!])
    end
    nothing
end

function _pyjl_reduce(self::C.PyPtr, ::C.PyPtr)
    v = _pyjl_serialize(self, C.PyNULL)
    v == C.PyNULL && return C.PyNULL
    args = C.PyTuple_New(1)
    args == C.PyNULL && (C.Py_DecRef(v); return C.PyNULL)
    err = C.PyTuple_SetItem(args, 0, v)
    err == -1 && (C.Py_DecRef(args); return C.PyNULL)
    red = C.PyTuple_New(2)
    red == C.PyNULL && (C.Py_DecRef(args); return C.PyNULL)
    err = C.PyTuple_SetItem(red, 1, args)
    err == -1 && (C.Py_DecRef(red); return C.PyNULL)
    f = C.PyObject_GetAttrString(self, "_jl_deserialize")
    f == C.PyNULL && (C.Py_DecRef(red); return C.PyNULL)
    err = C.PyTuple_SetItem(red, 0, f)
    err == -1 && (C.Py_DecRef(red); return C.PyNULL)
    return red
end

function _pyjl_serialize(self::C.PyPtr, ::C.PyPtr)
    try
        io = IOBuffer()
        serialize(io, PyJuliaValue_GetValue(self))
        b = take!(io)
        return C.PyBytes_FromStringAndSize(pointer(b), sizeof(b))
    catch e
        C.PyErr_SetString(C.POINTERS.PyExc_Exception, "error serializing this value")
        # wrap sprint in another try-catch block to prevent this function from throwing
        try
            @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
        catch
        end
        return C.PyNULL
    end
end

function _pyjl_deserialize(t::C.PyPtr, v::C.PyPtr)
    try
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{C.Py_ssize_t}()
        err = C.PyBytes_AsStringAndSize(v, ptr, len)
        err == -1 && return C.PyNULL
        io = IOBuffer(unsafe_wrap(Array, Ptr{UInt8}(ptr[]), Int(len[])))
        x = deserialize(io)
        return PyJuliaValue_New(t, x)
    catch e
        C.PyErr_SetString(C.POINTERS.PyExc_Exception, "error deserializing this value")
        # wrap sprint in another try-catch block to prevent this function from throwing
        try
            @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
        catch
        end
        return C.PyNULL
    end
end

const _pyjlbase_name = "juliacall.ValueBase"
const _pyjlbase_type = fill(C.PyTypeObject())
const _pyjlbase_isnull_name = "_jl_isnull"
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_reduce_name = "__reduce__"
const _pyjlbase_serialize_name = "_jl_serialize"
const _pyjlbase_deserialize_name = "_jl_deserialize"
const _pyjlbase_methods = Vector{C.PyMethodDef}()
const _pyjlbase_as_buffer = fill(C.PyBufferProcs())

function init_c()
    empty!(_pyjlbase_methods)
    push!(
        _pyjlbase_methods,
        C.PyMethodDef(
            name = pointer(_pyjlbase_callmethod_name),
            meth = @cfunction(_pyjl_callmethod, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_VARARGS,
        ),
        C.PyMethodDef(
            name = pointer(_pyjlbase_isnull_name),
            meth = @cfunction(_pyjl_isnull, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_NOARGS,
        ),
        C.PyMethodDef(
            name = pointer(_pyjlbase_reduce_name),
            meth = @cfunction(_pyjl_reduce, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_NOARGS,
        ),
        C.PyMethodDef(
            name = pointer(_pyjlbase_serialize_name),
            meth = @cfunction(_pyjl_serialize, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_NOARGS,
        ),
        C.PyMethodDef(
            name = pointer(_pyjlbase_deserialize_name),
            meth = @cfunction(_pyjl_deserialize, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_O | C.Py_METH_CLASS,
        ),
        C.PyMethodDef(),
    )
    _pyjlbase_as_buffer[] = C.PyBufferProcs(
        get = @cfunction(_pyjl_get_buffer, Cint, (C.PyPtr, Ptr{C.Py_buffer}, Cint)),
        release = @cfunction(_pyjl_release_buffer, Cvoid, (C.PyPtr, Ptr{C.Py_buffer})),
    )
    _pyjlbase_type[] = C.PyTypeObject(
        name = pointer(_pyjlbase_name),
        basicsize = sizeof(PyJuliaValueObject),
        # new = C.POINTERS.PyType_GenericNew,
        new = @cfunction(_pyjl_new, C.PyPtr, (C.PyPtr, C.PyPtr, C.PyPtr)),
        dealloc = @cfunction(_pyjl_dealloc, Cvoid, (C.PyPtr,)),
        flags = C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG,
        weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
        # getattro = C.POINTERS.PyObject_GenericGetAttr,
        # setattro = C.POINTERS.PyObject_GenericSetAttr,
        methods = pointer(_pyjlbase_methods),
        as_buffer = pointer(_pyjlbase_as_buffer),
    )
    o = PyJuliaBase_Type[] = C.PyPtr(pointer(_pyjlbase_type))
    if C.PyType_Ready(o) == -1
        C.PyErr_Print()
        error("Error initializing 'juliacall.ValueBase'")
    end
end

function __init__()
    init_c()
end

PyJuliaValue_IsNull(o) = Base.GC.@preserve o UnsafePtr{PyJuliaValueObject}(C.asptr(o)).value[] == 0

PyJuliaValue_GetValue(o) = Base.GC.@preserve o begin
    idx = UnsafePtr{PyJuliaValueObject}(C.asptr(o)).value[]
    Base.@lock PYJLVALUES_LOCK begin
        PYJLVALUES[idx]
    end
end

PyJuliaValue_SetValue(_o, @nospecialize(v)) = Base.GC.@preserve _o begin
    o = C.asptr(_o)
    idx = UnsafePtr{PyJuliaValueObject}(o).value[]
    if idx == 0
        Base.@lock PYJLVALUES_LOCK begin
            if isempty(PYJLFREEVALUES)
                idx = PYJLVALUES_NEXT_IDX[]
                PYJLVALUES_NEXT_IDX[] += 1
            else
                idx = pop!(PYJLFREEVALUES)
            end
            PYJLVALUES[idx] = v
        end
        UnsafePtr{PyJuliaValueObject}(o).value[] = idx
    else
        Base.@lock PYJLVALUES_LOCK begin
            PYJLVALUES[idx] = v
        end
    end
    nothing
end

PyJuliaValue_New(_t, @nospecialize(v)) = Base.GC.@preserve _t begin
    t = C.asptr(_t)
    if C.PyType_IsSubtype(t, PyJuliaBase_Type[]) != 1
        C.PyErr_SetString(
            C.POINTERS.PyExc_TypeError,
            "Expecting a subtype of 'juliacall.ValueBase'",
        )
        return C.PyNULL
    end
    o = C.PyObject_CallObject(t, C.PyNULL)
    o == C.PyNULL && return C.PyNULL
    PyJuliaValue_SetValue(o, v)
    return o
end

end
