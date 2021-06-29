const PYJLGCCACHE = Dict{PyPtr, Base.RefValue{Any}}()

function _pyjl_new(t::PyPtr, ::PyPtr, ::PyPtr)
    o = ccall(UnsafePtr{PyTypeObject}(t).alloc[!], PyPtr, (PyPtr, Py_ssize_t), t, 0)
    o == PyNULL && return PyNULL
    UnsafePtr{PyJuliaValueObject}(o).weaklist[] = PyNULL
    UnsafePtr{PyJuliaValueObject}(o).value[] = C_NULL
    return o
end

function _pyjl_dealloc(o::PyPtr)
    delete!(PYJLGCCACHE, o)
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

const _pyjlbase_name = "juliacall.ValueBase"
const _pyjlbase_type = fill(C.PyTypeObject())
const _pyjlbase_isnull_name = "_jl_isnull"
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_methods = Vector{PyMethodDef}()

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
        PyMethodDef(),
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
        methods = pointer(_pyjlbase_methods)
    )
    o = POINTERS.PyJuliaBase_Type = PyPtr(pointer(_pyjlbase_type))
    if PyType_Ready(o) == -1
        PyErr_Print()
        error("Error initializing 'juliacall.ValueBase'")
    end
end

PyJuliaValue_IsNull(o::PyPtr) = UnsafePtr{PyJuliaValueObject}(o).value[!] == C_NULL

PyJuliaValue_GetValue(o::PyPtr) = (Base.unsafe_pointer_to_objref(UnsafePtr{PyJuliaValueObject}(o).value[!])::Base.RefValue{Any})[]

PyJuliaValue_SetValue(o::PyPtr, v) = begin
    ref = Base.RefValue{Any}(v)
    PYJLGCCACHE[o] = ref
    UnsafePtr{PyJuliaValueObject}(o).value[] = Base.pointer_from_objref(ref)
    nothing
end

PyJuliaValue_New(t::PyPtr, v) = begin
    if PyType_IsSubtype(t, POINTERS.PyJuliaBase_Type) != 1
        PyErr_SetString(POINTERS.PyExc_TypeError, "Expecting a subtype of 'juliacall.ValueBase'")
        return PyNULL
    end
    o = PyObject_CallObject(t, PyNULL)
    o == PyNULL && return PyNULL
    PyJuliaValue_SetValue(o, v)
    return o
end
