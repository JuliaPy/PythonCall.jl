const PyJuliaDictValue_Type__ref = Ref(PyPtr())
PyJuliaDictValue_Type() = begin
    ptr = PyJuliaDictValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.DictValue",
            base = base,
            iter = pyjldict_iter,
            methods = [
                (name="keys", flags=Py_METH_NOARGS, meth=pyjldict_keys),
                (name="values", flags=Py_METH_NOARGS, meth=pyjldict_values),
                (name="items", flags=Py_METH_NOARGS, meth=pyjldict_items),
                (name="get", flags=Py_METH_VARARGS, meth=pyjldict_get),
            ],
            as_sequence = (
                contains = pyjldict_contains,
            ),
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyMutableMappingABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaDictValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaDictValue_New(x::AbstractDict) = PyJuliaValue_New(PyJuliaDictValue_Type(), x)
PyJuliaValue_From(x::AbstractDict) = PyJuliaDictValue_New(x)

pyjldict_iter(xo::PyPtr) = PyJuliaIteratorValue_New(Iterator(keys(PyJuliaValue_GetValue(xo)::AbstractDict)))

pyjldict_keys(xo::PyPtr, ::PyPtr) = PyObject_From(keys(PyJuliaValue_GetValue(xo)))
pyjldict_values(xo::PyPtr, ::PyPtr) = PyObject_From(values(PyJuliaValue_GetValue(xo)))
pyjldict_items(xo::PyPtr, ::PyPtr) = PyObject_From(DictPairSet(pairs(PyJuliaValue_GetValue(xo))))

pyjldict_contains(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    r = PyObject_TryConvert(vo, keytype(x))
    r == -1 && return Cint(-1)
    r ==  0 && return Cint(0)
    v = takeresult(keytype(x))
    try
        Cint(haskey(x, v))
    catch err
        PyErr_SetJuliaError(err)
        Cint(-1)
    end
end

pyjldict_get(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    ism1(PyArg_CheckNumArgsBetween("get", args, 1, 2)) && return PyPtr()
    ko = PyTuple_GetItem(args, 0)
    vo = PyTuple_Size(args) < 2 ? Py_None() : PyTuple_GetItem(args, 1)
    r = PyObject_TryConvert(ko, keytype(x))
    r == -1 && return PyPtr()
    r ==  0 && (Py_IncRef(vo); return vo)
    k = takeresult(keytype(x))
    try
        if haskey(x, k)
            PyObject_From(x[k])
        else
            Py_IncRef(vo)
            vo
        end
    catch err
        PyErr_SetJuliaError(err)
        Cint(-1)
    end
end

struct DictPairSet{K,V,T<:AbstractDict{K,V}} <: AbstractSet{Tuple{K,V}}
    dict :: T
end
Base.length(x::DictPairSet) = length(x.dict)
Base.iterate(x::DictPairSet) = (r=iterate(x.dict); r===nothing ? nothing : (Tuple(r[1]), r[2]))
Base.iterate(x::DictPairSet, st) = (r=iterate(x.dict, st); r===nothing ? nothing : (Tuple(r[1]), r[2]))
Base.in(v, x::DictPairSet) = v in x.dict
