asptr(x) = Base.unsafe_convert(PyPtr, x)

# Free-threaded CPython builds ("3.14t") currently have different C struct layouts,
# but there is no stable ABI yet. To keep the code manageable, we centralize the
# branching in a single macro that rewrites type names in the expression.
const _FT_TYPE_REPLACEMENTS = Dict{Symbol,Symbol}(
    :PyObject => :PyObjectFT,
    :PyVarObject => :PyVarObjectFT,
    :PyMemoryViewObject => :PyMemoryViewObjectFT,
    :PySimpleObject => :PySimpleObjectFT,
    # Used from JlWrap/C.jl via `C.@ft`.
    :PyJuliaValueObject => :PyJuliaValueObjectFT,
)

function _ft_transform(ex)
    if ex isa Symbol
        return get(_FT_TYPE_REPLACEMENTS, ex, ex)
    elseif ex isa QuoteNode
        return QuoteNode(_ft_transform(ex.value))
    elseif ex isa Expr
        return Expr(ex.head, map(_ft_transform, ex.args)...)
    else
        return ex
    end
end

"""
    @ft expr

Evaluate `expr`, but when `CTX.is_free_threaded` is true (CPython "free-threaded"
builds), rewrite internal type names like `PyObject` → `PyObjectFT` inside the
expression.

This keeps free-threaded branching centralized, so we don't scatter `if
CTX.is_free_threaded` throughout the code.
"""
macro ft(ex)
    ex_ft = _ft_transform(ex)
    m = @__MODULE__
    return esc(:($m.CTX.is_free_threaded ? $ex_ft : $ex))
end

Py_Type(x) = Base.GC.@preserve x @ft PyPtr(UnsafePtr{PyObject}(asptr(x)).type[!])

PyObject_Type(x) = Base.GC.@preserve x (t = Py_Type(asptr(x)); Py_IncRef(t); t)

Py_TypeCheck(o, t) = Base.GC.@preserve o t PyType_IsSubtype(Py_Type(asptr(o)), asptr(t))
Py_TypeCheckFast(o, f::Integer) = Base.GC.@preserve o PyType_IsSubtypeFast(Py_Type(asptr(o)), f)

PyType_IsSubtypeFast(t, f::Integer) =
    Base.GC.@preserve t Cint(!iszero(PyType_GetFlags(asptr(t)) & f))

PyMemoryView_GET_BUFFER(m) = Base.GC.@preserve m @ft Ptr{Py_buffer}(UnsafePtr{PyMemoryViewObject}(asptr(m)).view)

PyType_CheckBuffer(t) = Base.GC.@preserve t begin
    getbuf = PyType_GetSlot(asptr(t), Py_bf_getbuffer)
    return getbuf != C_NULL
end

PyObject_CheckBuffer(o) = Base.GC.@preserve o PyType_CheckBuffer(Py_Type(asptr(o)))

PyObject_GetBuffer(_o, b, flags) = Base.GC.@preserve _o begin
    o = asptr(_o)
    getbuf = PyType_GetSlot(Py_Type(o), Py_bf_getbuffer)
    if getbuf == C_NULL
        # TODO: we can drop this branch and just use PyType_GetName once we stop
        # supporting python 3.10
        msg = if CTX.is_free_threaded
            "a bytes-like object is required"
        else
            "a bytes-like object is required, not '$(String(UnsafePtr{PyTypeObject}(Py_Type(o)).name[]))'"
        end
        PyErr_SetString(POINTERS.PyExc_TypeError, msg)
        return Cint(-1)
    end
    return ccall(getbuf, Cint, (PyPtr, Ptr{Py_buffer}, Cint), o, b, flags)
end

PyBuffer_Release(_b) = begin
    b = UnsafePtr(Base.unsafe_convert(Ptr{Py_buffer}, _b))
    o = b.obj[]
    o == C_NULL && return
    releasebuf = PyType_GetSlot(Py_Type(o), Py_bf_releasebuffer)
    if releasebuf != C_NULL
        ccall(releasebuf, Cvoid, (PyPtr, Ptr{Py_buffer}), o, b)
    end
    b.obj[] = C_NULL
    Py_DecRef(o)
    return
end

function PyOS_SetInputHook(hook::Ptr{Cvoid})
    Base.unsafe_store!(POINTERS.PyOS_InputHookPtr, hook)
    return
end

function PyOS_GetInputHook()
    return Base.unsafe_load(POINTERS.PyOS_InputHookPtr)
end

function PyOS_RunInputHook()
    hook = PyOS_GetInputHook()
    if hook == C_NULL
        return false
    else
        ccall(hook, Cint, ())
        return true
    end
end

function PySimpleObject_GetValue(::Type{T}, o) where {T}
    Base.GC.@preserve o @ft UnsafePtr{PySimpleObject{T}}(asptr(o)).value[!]
end

# FAST REFCOUNTING
#
# _Py_IncRef(o) = ccall(POINTERS.Py_IncRef, Cvoid, (PyPtr,), o)
# _Py_DecRef(o) = ccall(POINTERS.Py_DecRef, Cvoid, (PyPtr,), o)
# const FAST_INCREF = true
# const FAST_DECREF = true
# if FAST_INCREF
#     # This avoids calling the C-API Py_IncRef().
#     # It just needs to increase the reference count.
#     # Assumes Python is not built for debugging reference counts.
#     # Speed up from 2.5ns to 1.3ns.
#     Py_INCREF(o) = GC.@preserve o begin
#         p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
#         p.refcnt[] += 1
#         nothing
#     end
#     Py_IncRef(o) = GC.@preserve o begin
#         p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
#         if p != C_NULL
#             p.refcnt[] += 1
#         end
#         nothing
#     end
# else
#     Py_INCREF(o) = _Py_IncRef(o)
#     Py_IncRef(o) = _Py_IncRef(o)
# end
# if FAST_DECREF
#     # This avoids calling the C-API Py_IncRef() unless the object is about to be deallocated.
#     # It just needs to decrement the reference count.
#     # Assumes Python is not built for debugging reference counts.
#     # Speed up from 2.5ns to 1.8ns in non-deallocating case.
#     Py_DECREF(o) = GC.@preserve o begin
#         p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
#         c = p.refcnt[]
#         if c > 1
#             p.refcnt[] = c - 1
#         else
#             _Py_DecRef(o)
#         end
#         nothing
#     end
#     Py_DecRef(o) = GC.@preserve o begin
#         p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
#         if p != C_NULL
#             c = p.refcnt[]
#             if c > 1
#                 p.refcnt[] = c - 1
#             else
#                 _Py_DecRef(o)
#             end
#         end
#         nothing
#     end
# else
#     Py_DECREF(o) = _Py_DecRef(o)
#     Py_DecRef(o) = _Py_DecRef(o)
# end
# Py_RefCnt(o) = GC.@preserve o UnsafePtr(Base.unsafe_convert(PyPtr, o)).refcnt[]

# Py_DecRef(f::Function, o::Ptr, dflt = PYERR()) =
#     isnull(o) ? dflt : (r = f(o); Py_DecRef(o); r)
