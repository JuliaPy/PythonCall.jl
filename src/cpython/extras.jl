Py_Type(x::PyPtr) = PyPtr(UnsafePtr(x).type[!])

PyObject_Type(x::PyPtr) = (t=Py_Type(x); Py_IncRef(t); t)

Py_TypeCheck(o::PyPtr, t::PyPtr) = PyType_IsSubtype(Py_Type(o), t)
Py_TypeCheckFast(o::PyPtr, f::Integer) = PyType_IsSubtypeFast(Py_Type(o), f)

PyType_IsSubtypeFast(t::PyPtr, f::Integer) = Cint(!iszero(UnsafePtr{PyTypeObject}(t).flags[] & f))

PyMemoryView_GET_BUFFER(m::PyPtr) = Ptr{Py_buffer}(UnsafePtr{PyMemoryViewObject}(m).view)

PyType_CheckBuffer(t::PyPtr) = begin
    p = UnsafePtr{PyTypeObject}(t).as_buffer[]
    return p != C_NULL && p.get[!] != C_NULL
end

PyObject_CheckBuffer(o::PyPtr) = PyType_CheckBuffer(Py_Type(o))

PyObject_GetBuffer(o::PyPtr, b, flags) = begin
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if p == C_NULL || p.get[!] == C_NULL
        PyErr_SetString(
            POINTERS.PyExc_TypeError,
            "a bytes-like object is required, not '$(String(UnsafePtr{PyTypeObject}(Py_Type(o)).name[]))'",
        )
        return Cint(-1)
    end
    return ccall(p.get[!], Cint, (PyPtr, Ptr{Py_buffer}, Cint), o, b, flags)
end

PyBuffer_Release(_b) = begin
    b = UnsafePtr(Base.unsafe_convert(Ptr{Py_buffer}, _b))
    o = b.obj[]
    o == C_NULL && return
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if (p != C_NULL && p.release[!] != C_NULL)
        ccall(p.release[!], Cvoid, (PyPtr, Ptr{Py_buffer}), o, b)
    end
    b.obj[] = C_NULL
    Py_DecRef(o)
    return
end

function PyOS_RunInputHook()
    hook = Base.unsafe_load(Ptr{Ptr{Cvoid}}(dlsym(CTX.lib_ptr, :PyOS_InputHook)))
    if hook == C_NULL
        return false
    else
        ccall(hook, Cint, ())
        return true
    end
end

function PySimpleObject_GetValue(::Type{T}, o::PyPtr) where {T}
    UnsafePtr{PySimpleObject{T}}(o).value[!]
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
