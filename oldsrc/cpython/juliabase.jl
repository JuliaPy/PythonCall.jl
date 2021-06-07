const PYJLGCCACHE = Dict{PyPtr,Any}()

@kwdef struct PyJuliaValueObject
    ob_base::PyObject = PyObject()
    value::Ptr{Cvoid} = C_NULL
    weaklist::PyPtr = C_NULL
end

pyjlbase_new(t::PyPtr, ::PyPtr, ::PyPtr) = begin
    o = ccall(UnsafePtr{PyTypeObject}(t).alloc[!], PyPtr, (PyPtr, Py_ssize_t), t, 0)
    if !isnull(o)
        PyJuliaValue_SetValue(o, nothing)
        UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C_NULL
    end
    o
end

pyjlbase_init(o::PyPtr, args::PyPtr, kwargs::PyPtr) = begin
    ism1(PyArg_CheckNumArgsEq("__init__", args, 1)) && return Cint(-1)
    ism1(PyArg_CheckNoKwargs("__init__", kwargs)) && return Cint(-1)
    ism1(PyArg_GetArg(Any, "__init__", args, 0)) && return Cint(-1)
    PyJuliaValue_SetValue(o, takeresult(Any))
    Cint(0)
end

pyjlbase_dealloc(o::PyPtr) = begin
    delete!(PYJLGCCACHE, o)
    isnull(UnsafePtr{PyJuliaValueObject}(o).weaklist[!]) || PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{PyTypeObject}(Py_Type(o)).free[!], Cvoid, (PyPtr,), o)
    nothing
end

const PyJuliaBaseValue_Type = LazyPyObject() do
    c = []
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.ValueBase"),
        basicsize = sizeof(PyJuliaValueObject),
        new = pyglobal(:PyType_GenericNew),
        init = @cfunctionIOOO(pyjlbase_init),
        dealloc = @cfunctionVO(pyjlbase_dealloc),
        flags = Py_TPFLAGS_BASETYPE |
                Py_TPFLAGS_HAVE_VERSION_TAG |
                (CONFIG.isstackless ? Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
        weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
        getattro = pyglobal(:PyObject_GenericGetAttr),
        setattro = pyglobal(:PyObject_GenericSetAttr),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaValue_Check(o) = Py_TypeCheck(o, PyJuliaBaseValue_Type())

PyJuliaValue_GetValue(__o) = begin
    _o = Base.cconvert(PyPtr, __o)
    GC.@preserve _o begin
        o = Base.unsafe_convert(PyPtr, _o)
        p = UnsafePtr{PyJuliaValueObject}(o)
        Base.unsafe_pointer_to_objref(p.value[!])
    end
end

PyJuliaValue_SetValue(__o, v) = begin
    _o = Base.cconvert(PyPtr, __o)
    GC.@preserve _o begin
        o = Base.unsafe_convert(PyPtr, _o)
        p = UnsafePtr{PyJuliaValueObject}(o)
        p.value[!], PYJLGCCACHE[o] = PythonCall.pointer_from_obj(v)
    end
end

PyJuliaValue_New(t, v) = begin
    if isnull(t)
        if !PyErr_IsSet()
            PyErr_SetString(PyExc_Exception(), "Got NULL type with no error set")
        end
        return PyNULL
    end
    bt = PyJuliaBaseValue_Type()
    isnull(bt) && return PyNULL
    PyType_IsSubtype(t, bt) != 0 || (
        PyErr_SetString(PyExc_TypeError(), "Expecting a subtype of 'juliacall.ValueBase'"); return PyNULL
    )
    o = _PyObject_New(t)
    isnull(o) && return PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C_NULL
    PyJuliaValue_SetValue(o, v)
    o
end

PyJuliaBaseValue_New(v) = begin
    t = PyJuliaBaseValue_Type()
    isnull(t) && return PyNULL
    PyJuliaValue_New(t, v)
end

PyJuliaValue_TryConvert_any(o, ::Type{S}) where {S} = begin
    x = PyJuliaValue_GetValue(o)
    putresult(tryconvert(S, x))
end

macro pyjltry(body, errval, handlers...)
    handlercode = []
    finalcode = []
    for handler in handlers
        handler isa Expr && handler.head === :call && handler.args[1] == :(=>) || error("invalid handler: $handler (not a pair)")
        jt, pt = handler.args[2:end]
        if jt === :Finally
            push!(finalcode, esc(pt))
            break
        elseif jt === :OnErr
            push!(handlercode, esc(pt))
            break
        end
        if jt isa Expr && jt.head === :tuple
            args = jt.args[2:end]
            jt = jt.args[1]
        else
            args = []
        end
        if jt === :MethodError
            if length(args) == 0
                cond = :(err isa MethodError)
            elseif length(args) == 1
                cond = :(err isa MethodError && err.f === $(esc(args[1])))
            elseif length(args) == 2
                cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2]))))
            elseif length(args) == 3
                cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3]))))
            elseif length(args) == 4
                cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4]))))
            elseif length(args) == 5
                cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4])) || err.f === $(esc(args[5]))))
            elseif length(args) == 6
                cond = :(err isa MethodError && (err.f === $(esc(args[1])) || err.f === $(esc(args[2])) || err.f === $(esc(args[3])) || err.f === $(esc(args[4])) || err.f === $(esc(args[5])) || err.f === $(esc(args[6]))))
            else
                error("not implemented: more than 6 arguments to MethodError")
            end
        elseif jt === :UndefVarError
            if length(args) == 0
                cond = :(err isa UndefVarError)
            elseif length(args) == 1
                cond = :(err isa UndefVarError && err.var === $(esc(args[1])))
            else
                error("not implemented: more than 1 argument to UndefVarError")
            end
        elseif jt === :BoundsError
            if length(args) == 0
                cond = :(err isa BoundsError)
            elseif length(args) == 1
                cond = :(err isa BoundsError && err.a === $(esc(args[1])))
            else
                error("not implemented: more than 1 argument to BoundsError")
            end
        elseif jt === :KeyError
            if length(args) == 0
                cond = :(err isa KeyError)
            elseif length(args) == 1
                cond = :(err isa KeyError && err.key === $(esc(args[1])))
            elseif length(args) == 2
                cond = :(err isa KeyError && (err.key === $(esc(args[1])) || err.key === $(esc(args[2]))))
            else
                error("not implemented: more than 2 arguments to KeyError")
            end
        elseif jt === :ErrorException
            if length(args) == 0
                cond = :(err isa ErrorException)
            elseif length(args) == 1
                cond = :(err isa ErrorException && match($(args[1]), err.msg) !== nothing)
            else
                error("not implemented: more than 1 argument to ErrorException")
            end
        elseif jt === :Custom
            if length(args) == 1
                cond = esc(args[1])
            else
                error("expecting 1 argument to Custom")
            end
        else
            error("invalid handler: $handler (bad julia error type)")
        end
        if pt === :JuliaError
            seterr = :(PyErr_SetJuliaError(err))
        elseif pt === :NotImplemented
            seterr = :(return PyNotImplemented_New())
        elseif pt in (:TypeError, :ValueError, :AttributeError, :NotImplementedError, :IndexError, :KeyError)
            seterr = :(PyErr_SetStringFromJuliaError($(Symbol(:PyExc_, pt))(), err))
        else
            error("invalid handler: $handler (bad python error type)")
        end
        push!(handlercode, :($cond && ($seterr; return $(esc(errval)))))
    end
    quote
        try
            $(esc(body))
        catch err
            $(handlercode...)
            PyErr_SetJuliaError(err)
            return $(esc(errval))
        finally
            $(finalcode...)
        end
    end
end
