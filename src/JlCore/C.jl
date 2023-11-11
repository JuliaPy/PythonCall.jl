module JlC

using Base
using ...Utils: Utils
using ...C: C
using ...Core
using ...Core: getptr, incref, errset, errmatches, errclear, pyisstr, pystr_asstring, pyJuliaError, pyistuple
using ...Convert: pyconvert
using ..JlCore: pyjl
using Base: @kwdef
using UnsafePointers: UnsafePtr
# using Serialization: serialize, deserialize

@kwdef struct PyJuliaValueObject
    ob_base::C.PyObject = C.PyObject()
    value::Int = 0
    weaklist::C.PyPtr = C_NULL
end

const PyJl_Type = Ref(C.PyNULL)

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
    if idx != 0
        PYJLVALUES[idx] = nothing
        push!(PYJLFREEVALUES, idx)
    end
    UnsafePtr{PyJuliaValueObject}(o).weaklist[!] == C.PyNULL || C.PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{C.PyTypeObject}(C.Py_Type(o)).free[!], Cvoid, (C.PyPtr,), o)
    nothing
end

function _return(x::Py; del::Bool=false)
    ptr = incref(getptr(x))
    del && unsafe_pydel!(x)
    ptr
end

function _raise(exc)
    try
        errset(pyJuliaError, pytuple((pyjl(exc), pyjl(catch_backtrace()))))
    catch
        @debug "Julia exception" exc
        errset(pybulitins.Exception, "an error occurred while raising a Julia error")
    end
    nothing
end

function _getany(ptr::C.PyPtr)
    if PyJl_Check(ptr)
        PyJl_GetValue(ptr)
    else
        pyconvert(Any, pynew(incref(ptr)))
    end
end

function _pyjl_repr(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        buf = IOBuffer()
        io = IOContext(buf, :limit=>true, :displaysize=>(23, 80))
        show(io, MIME("text/plain"), x)
        str = String(take!(buf))
        sep = '\n' in str ? '\n' : ' '
        ans = pystr("Julia:$sep$str")
        _return(ans, del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_str(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        buf = IOBuffer()
        io = IOContext(buf, :limit=>true, :displaysize=>(23, 80))
        print(io, x)
        str = String(take!(buf))
        ans = pystr(str)
        _return(ans, del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_hash(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        mod(hash(x), C.Py_hash_t)::C.Py_hash_t
    catch exc
        _raise(exc)
        C.Py_hash_t(0)
    end
end

function _pyjl_getattr(xptr::C.PyPtr, kptr::C.PyPtr)
    try
        # first do the generic lookup
        vptr = C.PyObject_GenericGetAttr(xptr, kptr)
        if vptr != C.PyNULL || !errmatches(pybuiltins.AttributeError)
            return vptr
        end
        errclear()
        # get the attribute name
        ko = pynew(incref(kptr))
        if !pyisstr(ko)
            errset(pybuiltins.TypeError, "attribute name must be string, not '$(pytype(ko).__name__)'")
            return C.PyNULL
        end
        kstr = pystr_asstring(ko)
        # skip attributes starting with "__" or "jl_"
        if startswith(kstr, "__") || startswith(kstr, "jl_")
            errset(pybuiltins.AttributeError, ko)
            unsafe_pydel!(ko)
            return C.PyNULL
        end
        unsafe_pydel!(ko)
        # get the property
        x = PyJl_GetValue(xptr)
        k = Symbol(kstr)
        v = getproperty(x, k)
        JlC.PyJl_New(v)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_setattr(xptr::C.PyPtr, kptr::C.PyPtr, vptr::C.PyPtr)
    try
        # first do the generic lookup
        err = C.PyObject_GenericSetAttr(xptr, kptr, vptr)
        if iszero(err) || !errmatches(pybuiltins.AttributeError)
            return err
        end
        errclear()
        # get the attribute name
        ko = pynew(incref(kptr))
        if !pyisstr(ko)
            errset(pybuiltins.TypeError, "attribute name must be string, not '$(pytype(ko).__name__)'")
            return Cint(-1)
        end
        kstr = pystr_asstring(ko)
        # skip attributes starting with "__" or "jl_"
        if startswith(kstr, "__") || startswith(kstr, "jl_")
            errset(pybuiltins.AttributeError, ko)
            unsafe_pydel!(ko)
            return Cint(-1)
        end
        unsafe_pydel!(ko)
        # set the property
        x = PyJl_GetValue(xptr)
        k = Symbol(kstr)
        v = _getany(vptr)
        setproperty!(x, k, v)
        Cint(0)
    catch exc
        _raise(exc)
        Cint(-1)
    end    
end

function _pyjl_len(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        v = length(x)
        convert(C.Py_ssize_t, v)
    catch exc
        _raise(exc)
        C.Py_ssize_t(-1)
    end
end

function _pyjl_getitem(xptr::C.PyPtr, kptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        if C.PyTuple_Check(kptr)
            k = pyconvert(Vector{Any}, pynew(incref(vptr)))
            v = x[k...]
        else
            k = _getany(kptr)
            v = x[k]
        end
        PyJl_New(v)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_setitem(xptr::C.PyPtr, kptr::C.PyPtr, vptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        v = _getany(vptr)
        if C.PyTuple_Check(kptr)
            k = pyconvert(Vector{Any}, pynew(incref(vptr)))
            x[k...] = v
        else
            k = _getany(kptr)
            x[k] = v
        end
        Cint(0)
    catch exc
        _raise(exc)
        Cint(-1)
    end
end

struct _pyjl_unary_op{F}
    op::F
end
function (f::_pyjl_unary_op)(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        v = f.op(x)
        _return(pyjl(v), del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

struct _pyjl_binary_op{F}
    op::F
end
function (f::_pyjl_binary_op)(xptr::C.PyPtr, yptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        y = _getany(yptr)
        v = f.op(x, y)
        _return(pyjl(v), del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_bool(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        if x isa Bool
            Cint(x)
        else
            errset(pybuiltins.TypeError, "only Julia 'Bool' values can be checked for truthyness, not '$(typeof(x))'")
            Cint(-1)
        end
    catch exc
        _raise(exc)
        Cint(-1)
    end
end

function _pyjl_int(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        _return(pyint(convert(Integer, x)), del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_index(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        if x isa Integer
            _return(pyint(x), del=true)
        else
            errset(pybuiltins.TypeError, "only Julia 'Integer' values can be used for indexing, not '$(typeof(x))'")
            C.PyNULL
        end
    catch exc
        _raise(exc)
        C.PyNULL
    end
end

function _pyjl_float(xptr::C.PyPtr)
    try
        x = PyJl_GetValue(xptr)
        _return(pyfloat(convert(Cdouble, x)), del=true)
    catch exc
        _raise(exc)
        C.PyNULL
    end
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

PyBufferInfo(::Any) = nothing

# TODO: the full implementation
PyBufferInfo(x::Array{Cdouble,N}) where {N} = PyBufferInfo{N}(ptr=Ptr{Cvoid}(pointer(x)), readonly=false, itemsize=sizeof(Cdouble), format="d", shape=size(x), strides=strides(x) .* sizeof(Cdouble))

function _pyjl_get_buffer_impl(obj::C.PyPtr, buf::Ptr{C.Py_buffer}, flags::Cint, info::PyBufferInfo{N}) where {N}
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
        C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "not C contiguous and strides not requested")
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
        C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "indirect array and suboffsets not requested")
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
    try
        info = PyBufferInfo(PyJl_GetValue(o))
        info === nothing && return Cint(-1)
        return _pyjl_get_buffer_impl(o, buf, flags, info::PyBufferInfo)::Cint
    catch exc
        @debug "error getting the buffer"
        C.PyErr_SetString(C.POINTERS.PyExc_BufferError, "some error occurred getting the buffer")
        return Cint(-1)
    end
end

function _pyjl_release_buffer(::C.PyPtr, buf::Ptr{C.Py_buffer})
    delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[!])
    nothing
end

# function _pyjl_reduce(self::C.PyPtr, ::C.PyPtr)
#     v = _pyjl_serialize(self, C.PyNULL)
#     v == C.PyNULL && return C.PyNULL
#     args = C.PyTuple_New(1)
#     args == C.PyNULL && (C.Py_DecRef(v); return C.PyNULL)
#     err = C.PyTuple_SetItem(args, 0, v)
#     err == -1 && (C.Py_DecRef(args); return C.PyNULL)
#     red = C.PyTuple_New(2)
#     red == C.PyNULL && (C.Py_DecRef(args); return C.PyNULL)
#     err = C.PyTuple_SetItem(red, 1, args)
#     err == -1 && (C.Py_DecRef(red); return C.PyNULL)
#     f = C.PyObject_GetAttrString(self, "_jl_deserialize")
#     f == C.PyNULL && (C.Py_DecRef(red); return C.PyNULL)
#     err = C.PyTuple_SetItem(red, 0, f)
#     err == -1 && (C.Py_DecRef(red); return C.PyNULL)
#     return red
# end

# function _pyjl_serialize(self::C.PyPtr, ::C.PyPtr)
#     try
#         io = IOBuffer()
#         serialize(io, PyJl_GetValue(self))
#         b = take!(io)
#         return C.PyBytes_FromStringAndSize(pointer(b), sizeof(b))
#     catch e
#         C.PyErr_SetString(C.POINTERS.PyExc_Exception, "error serializing this value")
#         # wrap sprint in another try-catch block to prevent this function from throwing
#         try
#             @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
#         catch
#         end
#         return C.PyNULL
#     end
# end

# function _pyjl_deserialize(t::C.PyPtr, v::C.PyPtr)
#     try
#         ptr = Ref{Ptr{Cchar}}()
#         len = Ref{C.Py_ssize_t}()
#         err = C.PyBytes_AsStringAndSize(v, ptr, len)
#         err == -1 && return C.PyNULL
#         io = IOBuffer(unsafe_wrap(Array, Ptr{UInt8}(ptr[]), Int(len[])))
#         x = deserialize(io)
#         return PyJl_New(t, x)
#     catch e
#         C.PyErr_SetString(C.POINTERS.PyExc_Exception, "error deserializing this value")
#         # wrap sprint in another try-catch block to prevent this function from throwing
#         try
#             @debug "Caught exception $(sprint(showerror, e, catch_backtrace()))"
#         catch
#         end
#         return C.PyNULL
#     end
# end

const _pyjl_name = "juliacall.Jl"
const _pyjl_type = fill(C.PyTypeObject())
# const _pyjl_isnull_name = "_jl_isnull"
# const _pyjl_callmethod_name = "_jl_callmethod"
# const _pyjl_reduce_name = "__reduce__"
# const _pyjl_serialize_name = "_jl_serialize"
# const _pyjl_deserialize_name = "_jl_deserialize"
const _pyjl_methods = Vector{C.PyMethodDef}()
const _pyjl_as_buffer = fill(C.PyBufferProcs())
const _pyjl_as_number = fill(C.PyNumberMethods())
const _pyjl_as_sequence = fill(C.PySequenceMethods())
const _pyjl_as_mapping = fill(C.PyMappingMethods())

function init_pyjl()
    empty!(_pyjl_methods)
    push!(_pyjl_methods,
        # C.PyMethodDef(
        #     name = pointer(_pyjl_callmethod_name),
        #     meth = @cfunction(_pyjl_callmethod, C.PyPtr, (C.PyPtr, C.PyPtr)),
        #     flags = C.Py_METH_VARARGS,
        # ),
        # C.PyMethodDef(
        #     name = pointer(_pyjl_isnull_name),
        #     meth = @cfunction(_pyjl_isnull, C.PyPtr, (C.PyPtr, C.PyPtr)),
        #     flags = C.Py_METH_NOARGS,
        # ),
        # C.PyMethodDef(
        #     name = pointer(_pyjl_reduce_name),
        #     meth = @cfunction(_pyjl_reduce, C.PyPtr, (C.PyPtr, C.PyPtr)),
        #     flags = C.Py_METH_NOARGS,
        # ),
        # C.PyMethodDef(
        #     name = pointer(_pyjl_serialize_name),
        #     meth = @cfunction(_pyjl_serialize, C.PyPtr, (C.PyPtr, C.PyPtr)),
        #     flags = C.Py_METH_NOARGS,
        # ),
        # C.PyMethodDef(
        #     name = pointer(_pyjl_deserialize_name),
        #     meth = @cfunction(_pyjl_deserialize, C.PyPtr, (C.PyPtr, C.PyPtr)),
        #     flags = C.Py_METH_O | C.Py_METH_CLASS,
        # ),
        C.PyMethodDef(),
    )
    _pyjl_as_number[] = C.PyNumberMethods(
        add = @cfunction(_pyjl_binary_op(+), C.PyPtr, (C.PyPtr, C.PyPtr)),
        subtract = @cfunction(_pyjl_binary_op(-), C.PyPtr, (C.PyPtr, C.PyPtr)),
        multiply = @cfunction(_pyjl_binary_op(*), C.PyPtr, (C.PyPtr, C.PyPtr)),
        remainder = @cfunction(_pyjl_binary_op(%), C.PyPtr, (C.PyPtr, C.PyPtr)),
        lshift = @cfunction(_pyjl_binary_op(<<), C.PyPtr, (C.PyPtr, C.PyPtr)),
        rshift = @cfunction(_pyjl_binary_op(>>), C.PyPtr, (C.PyPtr, C.PyPtr)),
        and = @cfunction(_pyjl_binary_op(&), C.PyPtr, (C.PyPtr, C.PyPtr)),
        xor = @cfunction(_pyjl_binary_op(⊻), C.PyPtr, (C.PyPtr, C.PyPtr)),
        or = @cfunction(_pyjl_binary_op(|), C.PyPtr, (C.PyPtr, C.PyPtr)),
        floordivide = @cfunction(_pyjl_binary_op(÷), C.PyPtr, (C.PyPtr, C.PyPtr)),
        truedivide = @cfunction(_pyjl_binary_op(/), C.PyPtr, (C.PyPtr, C.PyPtr)),
        negative = @cfunction(_pyjl_unary_op(-), C.PyPtr, (C.PyPtr,)),
        positive = @cfunction(_pyjl_unary_op(+), C.PyPtr, (C.PyPtr,)),
        absolute = @cfunction(_pyjl_unary_op(abs), C.PyPtr, (C.PyPtr,)),
        invert = @cfunction(_pyjl_unary_op(-), C.PyPtr, (C.PyPtr,)),
        bool = @cfunction(_pyjl_bool, Cint, (C.PyPtr,)),
        int = @cfunction(_pyjl_int, C.PyPtr, (C.PyPtr,)),
        index = @cfunction(_pyjl_index, C.PyPtr, (C.PyPtr,)),
        float = @cfunction(_pyjl_float, C.PyPtr, (C.PyPtr,)),
    )
    _pyjl_as_sequence[] = C.PySequenceMethods(
        # TODO
    )
    _pyjl_as_mapping[] = C.PyMappingMethods(
        length = @cfunction(_pyjl_len, C.Py_ssize_t, (C.PyPtr,)),
        subscript = @cfunction(_pyjl_getitem, C.PyPtr, (C.PyPtr, C.PyPtr)),
        ass_subscript = @cfunction(_pyjl_setitem, Cint, (C.PyPtr, C.PyPtr, C.PyPtr)),
    )
    _pyjl_as_buffer[] = C.PyBufferProcs(
        get = @cfunction(_pyjl_get_buffer, Cint, (C.PyPtr, Ptr{C.Py_buffer}, Cint)),
        release = @cfunction(_pyjl_release_buffer, Cvoid, (C.PyPtr, Ptr{C.Py_buffer})),
    )
    _pyjl_type[] = C.PyTypeObject(
        name = pointer(_pyjl_name),
        basicsize = sizeof(PyJuliaValueObject),
        # new = C.POINTERS.PyType_GenericNew,
        new = @cfunction(_pyjl_new, C.PyPtr, (C.PyPtr, C.PyPtr, C.PyPtr)),
        dealloc = @cfunction(_pyjl_dealloc, Cvoid, (C.PyPtr,)),
        flags = C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG,
        weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
        getattro = @cfunction(_pyjl_getattr, C.PyPtr, (C.PyPtr, C.PyPtr)),
        setattro = @cfunction(_pyjl_setattr, Cint, (C.PyPtr, C.PyPtr, C.PyPtr)),
        methods = pointer(_pyjl_methods),
        as_number = pointer(_pyjl_as_number),
        as_sequence = pointer(_pyjl_as_sequence),
        as_mapping = pointer(_pyjl_as_mapping),
        as_buffer = pointer(_pyjl_as_buffer),
        repr = @cfunction(_pyjl_repr, C.PyPtr, (C.PyPtr,)),
        str = @cfunction(_pyjl_str, C.PyPtr, (C.PyPtr,)),
        hash = @cfunction(_pyjl_hash, C.Py_hash_t, (C.PyPtr,)),
    )
    o = PyJl_Type[] = C.PyPtr(pointer(_pyjl_type))
    if C.PyType_Ready(o) == -1
        C.PyErr_Print()
        error("Error initializing 'juliacall.ValueBase'")
    end
end

function __init__()
    C.with_gil() do 
        init_pyjl()
    end
end

PyJl_GetIndex(o) = UnsafePtr{PyJuliaValueObject}(C.ptr(o)).value[]

PyJl_IsNew(o) = PyJl_GetIndex(o) == 0

function PyJl_GetValue(o)
    idx = PyJl_GetIndex(o)
    if idx == 0
        nothing
    else
        PYJLVALUES[idx]
    end
end

function PyJl_SetValue(o, @nospecialize(v))
    idx = PyJl_GetIndex(o)
    if idx == 0
        if isempty(PYJLFREEVALUES)
            push!(PYJLVALUES, v)
            idx = length(PYJLVALUES)
        else
            idx = pop!(PYJLFREEVALUES)
            PYJLVALUES[idx] = v
        end
        UnsafePtr{PyJuliaValueObject}(C.ptr(o)).value[] = idx
    else
        PYJLVALUES[idx] = v
    end
    nothing
end

function PyJl_New()
    C.PyObject_CallObject(PyJl_Type[], C.PyNULL)
end

function PyJl_New(@nospecialize(v))
    o = PyJl_New()
    o == C.PyNULL && return C.PyNULL
    PyJl_SetValue(o, v)
    o
end

function PyJl_Check(o)
    C.PyObject_IsInstance(o, PyJl_Type[]) == 1
end

end
