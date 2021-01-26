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

const PyJuliaBaseValue_Type__ref = Ref(PyPtr())
PyJuliaBaseValue_Type() = begin
    ptr = PyJuliaBaseValue_Type__ref[]
    if isnull(ptr)
        c = []
        t = fill(
            PyType_Create(
                c,
                name = "julia.ValueBase",
                basicsize = sizeof(PyJuliaValueObject),
                new = pyglobal(:PyType_GenericNew),
                init = @cfunction(pyjlbase_init, Cint, (PyPtr, PyPtr, PyPtr)),
                dealloc = @cfunction(pyjlbase_dealloc, Cvoid, (PyPtr,)),
                flags = Py_TPFLAGS_BASETYPE |
                        Py_TPFLAGS_HAVE_VERSION_TAG |
                        (CONFIG.isstackless ? Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
                weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
                getattro = pyglobal(:PyObject_GenericGetAttr),
                setattro = pyglobal(:PyObject_GenericSetAttr),
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaBaseValue_Type__ref[] = ptr
    end
    ptr
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
        p.value[!], PYJLGCCACHE[o] = Python.pointer_from_obj(v)
    end
end

PyJuliaValue_New(t, v) = begin
    if isnull(t)
        if !PyErr_IsSet()
            PyErr_SetString(PyExc_Exception(), "Got NULL type with no error set")
        end
        return PyPtr()
    end
    bt = PyJuliaBaseValue_Type()
    isnull(bt) && return PyPtr()
    PyType_IsSubtype(t, bt) != 0 || (
        PyErr_SetString(PyExc_TypeError(), "Expecting a subtype of 'julia.ValueBase'"); return PyPtr()
    )
    o = _PyObject_New(t)
    isnull(o) && return PyPtr()
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = C_NULL
    PyJuliaValue_SetValue(o, v)
    o
end

PyJuliaBaseValue_New(v) = begin
    t = PyJuliaBaseValue_Type()
    isnull(t) && return PyPtr()
    PyJuliaValue_New(t, v)
end

PyJuliaValue_TryConvert_any(o, ::Type{S}) where {S} = begin
    x = PyJuliaValue_GetValue(o)
    putresult(tryconvert(S, x))
end
