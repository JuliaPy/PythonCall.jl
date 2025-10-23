module Cjl

using ...C: C
using ...Utils: Utils
using ...Core: incref, pynew
using ...Convert: pyconvert
using Base: @kwdef
using UnsafePointers: UnsafePtr
using Serialization: serialize, deserialize

@kwdef struct PyJuliaValueObject
    ob_base::C.PyObject = C.PyObject()
    value::Int = 0
    weaklist::C.PyPtr = C_NULL
end

const PyJuliaBase_Type = Ref(C.PyNULL)
const PyJuliaBase_New = Ref(C.PyNULL)

# we store the actual julia values here
# the `value` field of `PyJuliaValueObject` indexes into here
const PYJLVALUES = []
# unused indices in PYJLVALUES
const PYJLFREEVALUES = Int[]

function _pyjl_new(t::C.PyPtr, ::C.PyPtr, ::C.PyPtr)
    o = ccall(UnsafePtr{C.PyTypeObject}(t).alloc[!], C.PyPtr, (C.PyPtr, C.Py_ssize_t), t, 0)
    o == C.PyNULL && return C.PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C.PyNULL
    UnsafePtr{PyJuliaValueObject}(o).value[] = 0
    return o
end

function _pyjl_dealloc(o::C.PyPtr)
    idx = UnsafePtr{PyJuliaValueObject}(o).value[]
    if idx >= 1
        PYJLVALUES[idx] = nothing
        push!(PYJLFREEVALUES, idx)
    end
    UnsafePtr{PyJuliaValueObject}(o).weaklist[!] == C.PyNULL || C.PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{C.PyTypeObject}(C.Py_Type(o)).free[!], Cvoid, (C.PyPtr,), o)
    nothing
end

function _getany(ptr::C.PyPtr)
    if PyJuliaValue_Check(ptr) == 1
        PyJuliaValue_GetValue(ptr)
    else
        pyconvert(Any, pynew(incref(ptr)))
    end
end

function _getany(::Type{T}, ptr::C.PyPtr) where {T}
    if PyJuliaValue_Check(ptr) == 1
        convert(T, PyJuliaValue_GetValue(ptr))::T
    else
        pyconvert(T, pynew(incref(ptr)))::T
    end
end

function _pyjl_init(xptr::C.PyPtr, argsptr::C.PyPtr, kwargsptr::C.PyPtr)
    if kwargsptr != C.PyNULL && C.PyDict_Size(kwargsptr) != 0
        errset(pybuiltins.TypeError, "keyword arguments not allowed")
        return Cint(-1)
    end
    if argsptr == C.PyNULL
        return Cint(0)
    end
    nargs = C.PyTuple_Size(argsptr)
    if nargs == 0
        return Cint(0)
    elseif nargs > 2
        errset(pybuiltins.TypeError, "__init__() takes up to 2 arguments ($nargs given)")
        return Cint(-1)
    end
    vptr = C.PyTuple_GetItem(argsptr, 0)
    try
        if nargs == 1
            v = _getany(vptr)
        else
            tptr = C.PyTuple_GetItem(argsptr, 1)
            t = _getany(tptr)
            if !isa(t, Type)
                C.PyErr_SetString(
                    C.POINTERS.PyExc_TypeError,
                    "type argument must be a Julia 'Type', not '$(typeof(t))'",
                )
                return Cint(-1)
            end
            v = _getany(t, vptr)
        end
        PyJuliaValue_SetValue(xptr, v)
        Cint(0)
    catch exc
        errtype =
            exc isa MethodError ? C.POINTERS.PyExc_TypeError : C.POINTERS.PyExc_Exception
        errmsg = sprint(showerror, exc)
        C.PyErr_SetString(errtype, errmsg)
        Cint(-1)
    end
end

const PYJLMETHODS = Vector{Any}()

function PyJulia_MethodNum(f)
    @nospecialize f
    push!(PYJLMETHODS, f)
    return length(PYJLMETHODS)
end

function _pyjl_callmethod(o::C.PyPtr, args::C.PyPtr)
    nargs = C.PyTuple_Size(args)
    @assert nargs > 0
    num = C.PyLong_AsLongLong(C.PyTuple_GetItem(args, 0))
    num == -1 && return C.PyNULL
    f = PYJLMETHODS[num]
    # this form gets defined in jlwrap/base.jl
    return _pyjl_callmethod(f, o, args, nargs)::C.PyPtr
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
    PYJLBUFCACHE[cptr] = c
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
        f = PYJLMETHODS[num]
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
    delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[!])
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

const _pyjlbase_name = "juliacall.JlBase"
const _pyjlbase_type = fill(C.PyTypeObject())
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_reduce_name = "__reduce__"
const _pyjlbase_serialize_name = "_jl_serialize"
const _pyjlbase_deserialize_name = "_jl_deserialize"
const _pyjlbase_weaklistoffset_name = "__weaklistoffset__"
const _pyjlbase_methods = Vector{C.PyMethodDef}()
const _pyjlbase_members = Vector{C.PyMemberDef}()
const _pyjlbase_slots = Vector{C.PyType_Slot}()
const _pyjlbase_spec = fill(C.PyType_Spec())

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

    # Create members for weakref support
    empty!(_pyjlbase_members)
    push!(
        _pyjlbase_members,
        C.PyMemberDef(
            name = pointer(_pyjlbase_weaklistoffset_name),
            typ = C.Py_T_PYSSIZET,
            offset = fieldoffset(PyJuliaValueObject, 3),
            flags = C.Py_READONLY,
        ),
        C.PyMemberDef(), # NULL terminator
    )

    # Create slots for PyType_Spec
    empty!(_pyjlbase_slots)
    push!(
        _pyjlbase_slots,
        C.PyType_Slot(
            slot = C.Py_tp_new,
            pfunc = @cfunction(_pyjl_new, C.PyPtr, (C.PyPtr, C.PyPtr, C.PyPtr))
        ),
        C.PyType_Slot(
            slot = C.Py_tp_dealloc,
            pfunc = @cfunction(_pyjl_dealloc, Cvoid, (C.PyPtr,))
        ),
        C.PyType_Slot(
            slot = C.Py_tp_init,
            pfunc = @cfunction(_pyjl_init, Cint, (C.PyPtr, C.PyPtr, C.PyPtr))
        ),
        C.PyType_Slot(slot = C.Py_tp_methods, pfunc = pointer(_pyjlbase_methods)),
        C.PyType_Slot(slot = C.Py_tp_members, pfunc = pointer(_pyjlbase_members)),
        C.PyType_Slot(
            slot = C.Py_bf_getbuffer,
            pfunc = @cfunction(_pyjl_get_buffer, Cint, (C.PyPtr, Ptr{C.Py_buffer}, Cint))
        ),
        C.PyType_Slot(
            slot = C.Py_bf_releasebuffer,
            pfunc = @cfunction(_pyjl_release_buffer, Cvoid, (C.PyPtr, Ptr{C.Py_buffer}))
        ),
        C.PyType_Slot(), # NULL terminator
    )

    # Create PyType_Spec
    _pyjlbase_spec[] = C.PyType_Spec(
        name = pointer(_pyjlbase_name),
        basicsize = sizeof(PyJuliaValueObject),
        flags = C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG,
        slots = pointer(_pyjlbase_slots),
    )

    # Create type using PyType_FromSpec
    o = PyJuliaBase_Type[] = C.PyType_FromSpec(pointer(_pyjlbase_spec))
    if o == C.PyNULL
        C.PyErr_Print()
        error("Error initializing 'juliacall.JlBase'")
    end
    n = PyJuliaBase_New[] = C.PyObject_GetAttrString(o, "__new__")
    if n == C.PyNULL
        C.PyErr_Print()
        error("Error accessing 'juliacall.JlBase.__new__'")
    end
    nothing
end

function __init__()
    init_c()
end

PyJuliaValue_Check(o) =
    Base.GC.@preserve o C.PyObject_IsInstance(C.asptr(o), PyJuliaBase_Type[])

PyJuliaValue_GetValue(o) = Base.GC.@preserve o begin
    v = UnsafePtr{PyJuliaValueObject}(C.asptr(o)).value[]
    if v == 0
        nothing
    elseif v > 0
        PYJLVALUES[v]
    elseif v == -1
        false
    elseif v == -2
        true
    end
end

PyJuliaValue_SetValue(o, v::Union{Nothing,Bool}) = Base.GC.@preserve o begin
    optr = UnsafePtr{PyJuliaValueObject}(C.asptr(o))
    idx = optr.value[]
    if idx >= 1
        PYJLVALUES[idx] = nothing
        push!(PYJLFREEVALUES, idx)
    end
    if v === nothing
        idx = 0
    elseif v === false
        idx = -1
    elseif v === true
        idx = -2
    else
        @assert false
    end
    optr.value[] = idx
    nothing
end

PyJuliaValue_SetValue(o, @nospecialize(v)) = Base.GC.@preserve o begin
    optr = UnsafePtr{PyJuliaValueObject}(C.asptr(o))
    idx = optr.value[]
    if idx >= 1
        PYJLVALUES[idx] = v
    else
        if isempty(PYJLFREEVALUES)
            push!(PYJLVALUES, v)
            idx = length(PYJLVALUES)
        else
            idx = pop!(PYJLFREEVALUES)
            PYJLVALUES[idx] = v
        end
        optr.value[] = idx
    end
    nothing
end

PyJuliaValue_New(t, @nospecialize(v)) = Base.GC.@preserve t begin
    tptr = C.asptr(t)
    if C.PyType_IsSubtype(tptr, PyJuliaBase_Type[]) != 1
        C.PyErr_SetString(
            C.POINTERS.PyExc_TypeError,
            "Expecting a subtype of 'juliacall.JlBase'",
        )
        return C.PyNULL
    end
    # All of this just to do JuliaBase.__new__(t). We do this to avoid calling `__init__`
    # which itself sets the value, and so duplicates work. Some classes such as `JlArray` do
    # not allow calling `__init__` with no args.
    # TODO: it could be replaced with PyObject_CallOneArg(PyJuliaBase_New[], t) when we drop
    # support for Python 3.8.
    args = C.PyTuple_New(1)
    args == C.PyNULL && return C.PyNULL
    C.Py_IncRef(tptr)
    err = C.PyTuple_SetItem(args, 0, tptr)
    err == -1 && (C.Py_DecRef(args); return C.PyNULL)
    o = C.PyObject_CallObject(PyJuliaBase_New[], args)
    C.Py_DecRef(args)
    o == C.PyNULL && return C.PyNULL
    PyJuliaValue_SetValue(o, v)
    return o
end

end
