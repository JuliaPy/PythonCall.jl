const PYJLGCCACHE = Dict{PyPtr, Any}()

_pyjl_new(::PyPtr, ::PyPtr, ::PyPtr) = begin
    PyErr_SetString(POINTERS.PyExc_TypeError, "'juliacall.ValueBase' objects cannot be created from Python")
    return PyNULL
end

_pyjl_dealloc(o::PyPtr) = begin
    delete!(PYJLGCCACHE, o)
    UnsafePtr{PyJuliaValueObject}(o).weaklist[!] == PyNULL || PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{C.PyTypeObject}(Py_Type(o)).free[!], Cvoid, (PyPtr,), o)
    nothing
end

const PYJLMETHODS = Vector{Function}()

function PyJulia_MethodNum(f::Function)
    @nospecialize f
    push!(PYJLMETHODS, f)
    return length(PYJLMETHODS)
end

_pyjl_callmethod(o::PyPtr, args::PyPtr) = begin
    nargs = PyTuple_Size(args)
    @assert nargs > 0
    num = PyLong_AsLongLong(PyTuple_GetItem(args, 0))
    num == -1 && return PyNULL
    f = PYJLMETHODS[num]
    # this form gets defined in jlwrap/base.jl
    return _pyjl_callmethod(f, o, args, nargs)::PyPtr
end

const _pyjlbase_name = "juliacall.ValueBase"
const _pyjlbase_type = fill(C.PyTypeObject())
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_methods = Vector{PyMethodDef}()

function init_jlwrap()
    push!(_pyjlbase_methods, PyMethodDef(
        name = pointer(_pyjlbase_callmethod_name),
        meth = @cfunction(_pyjl_callmethod, PyPtr, (PyPtr, PyPtr)),
        flags = Py_METH_VARARGS,
    ), PyMethodDef())
    t = UnsafePtr(_pyjlbase_type)
    t.name[] = pointer(_pyjlbase_name)
    t.basicsize[] = sizeof(PyJuliaValueObject)
    t.new[] = @cfunction(_pyjl_new, PyPtr, (PyPtr, PyPtr, PyPtr))
    t.dealloc[] = @cfunction(_pyjl_dealloc, Cvoid, (PyPtr,))
    t.flags[] = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_VERSION_TAG
    t.weaklistoffset[] = fieldoffset(PyJuliaValueObject, 3)
    t.getattro[] = POINTERS.PyObject_GenericGetAttr
    t.setattro[] = POINTERS.PyObject_GenericSetAttr
    t.methods[] = pointer(_pyjlbase_methods)
    o = POINTERS.PyJuliaBase_Type = PyPtr(t)
    if PyType_Ready(o) == -1
        PyErr_Print()
        error("Error initializing 'juliacall.ValueBase'")
    end
end

PyJuliaValue_GetValue(o::PyPtr) = Base.unsafe_pointer_to_objref(UnsafePtr{PyJuliaValueObject}(o).value[!])

PyJuliaValue_SetValue(o::PyPtr, v) = begin
    p = UnsafePtr{PyJuliaValueObject}(o)
    p.value[!], PYJLGCCACHE[o] = Utils.pointer_from_obj(v)
    nothing
end

PyJuliaValue_New(t::PyPtr, v) = begin
    if PyType_IsSubtype(t, POINTERS.PyJuliaBase_Type) != 1
        PyErr_SetString(POINTERS.TypeError, "Expecting a subtype of 'juliacall.ValueBase'")
        return PyNULL
    end
    o = _PyObject_New(t)
    o == PyNULL && return PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C_NULL
    PyJuliaValue_SetValue(o, v)
    return o
end


# PyJuliaValue_TryConvert_any(o, ::Type{S}) where {S} = begin
#     x = PyJuliaValue_GetValue(o)
#     putresult(tryconvert(S, x))
# end

# macro pyjltry(body, errval, handlers...)
#     handlercode = []
#     finalcode = []
#     for handler in handlers
#         handler isa Expr && handler.head === :call && handler.args[1] == :(=>) || error("invalid handler: $handler (not a pair)")
#         jt, pt = handler.args[2:end]
#         if jt === :Finally
#             push!(finalcode, esc(pt))
#             break
#         elseif jt === :OnErr
#             push!(handlercode, esc(pt))
#             break
#         end
#         if jt isa Expr && jt.head === :tuple
#             args = jt.args[2:end]
#             jt = jt.args[1]
#         else
#             args = []
#         end
#         if jt === :MethodError
#             if length(args) == 0
#                 cond = :(err isa MethodError)
#             elseif length(args) == 1
#                 cond = :(err isa MethodError && err.f === $(esc(args[1])))
#             elseif length(args) == 2
#                 cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2]))))
#             elseif length(args) == 3
#                 cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3]))))
#             elseif length(args) == 4
#                 cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4]))))
#             elseif length(args) == 5
#                 cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4])) || err.f === $(esc(args[5]))))
#             elseif length(args) == 6
#                 cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4])) || err.f === $(esc(args[5])) || err.f === $(esc(args[6]))))
#             else
#                 error("not implemented: more than 6 arguments to MethodError")
#             end
#         elseif jt === :UndefVarError
#             if length(args) == 0
#                 cond = :(err isa UndefVarError)
#             elseif length(args) == 1
#                 cond = :(err isa UndefVarError && err.var === $(esc(args[1])))
#             else
#                 error("not implemented: more than 1 argument to UndefVarError")
#             end
#         elseif jt === :BoundsError
#             if length(args) == 0
#                 cond = :(err isa BoundsError)
#             elseif length(args) == 1
#                 cond = :(err isa BoundsError && err.a === $(esc(args[1])))
#             else
#                 error("not implemented: more than 1 argument to BoundsError")
#             end
#         elseif jt === :KeyError
#             if length(args) == 0
#                 cond = :(err isa KeyError)
#             elseif length(args) == 1
#                 cond = :(err isa KeyError && err.key === $(esc(args[1])))
#             elseif length(args) == 2
#                 cond = :(err isa KeyError && (err.key === $(esc(args[1])) || err.key === $(esc(args[2]))))
#             else
#                 error("not implemented: more than 2 arguments to KeyError")
#             end
#         elseif jt === :ErrorException
#             if length(args) == 0
#                 cond = :(err isa ErrorException)
#             elseif length(args) == 1
#                 cond = :(err isa ErrorException && match($(args[1]), err.msg) !== nothing)
#             else
#                 error("not implemented: more than 1 argument to ErrorException")
#             end
#         elseif jt === :Custom
#             if length(args) == 1
#                 cond = esc(args[1])
#             else
#                 error("expecting 1 argument to Custom")
#             end
#         else
#             error("invalid handler: $handler (bad julia error type)")
#         end
#         if pt === :JuliaError
#             seterr = :(PyErr_SetJuliaError(err))
#         elseif pt === :NotImplemented
#             seterr = :(return PyNotImplemented_New())
#         elseif pt in (:TypeError, :ValueError, :AttributeError, :NotImplementedError, :IndexError, :KeyError)
#             seterr = :(PyErr_SetStringFromJuliaError($(Symbol(:PyExc_, pt))(), err))
#         else
#             error("invalid handler: $handler (bad python error type)")
#         end
#         push!(handlercode, :($cond && ($seterr; return $(esc(errval)))))
#     end
#     quote
#         try
#             $(esc(body))
#         catch err
#             $(handlercode...)
#             PyErr_SetJuliaError(err)
#             return $(esc(errval))
#         finally
#             $(finalcode...)
#         end
#     end
# end
