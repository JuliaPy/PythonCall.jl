pyjldict_iter(xo::PyPtr) =
    PyJuliaIteratorValue_New(Iterator(keys(PyJuliaValue_GetValue(xo)::AbstractDict)))

pyjldict_keys(xo::PyPtr, ::PyPtr) = PyObject_From(keys(PyJuliaValue_GetValue(xo)))
pyjldict_values(xo::PyPtr, ::PyPtr) = PyObject_From(values(PyJuliaValue_GetValue(xo)))
pyjldict_items(xo::PyPtr, ::PyPtr) =
    PyObject_From(DictPairSet(pairs(PyJuliaValue_GetValue(xo))))

pyjldict_contains(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    r = PyObject_TryConvert(vo, keytype(x))
    r == -1 && return Cint(-1)
    r == 0 && return Cint(0)
    v = takeresult(keytype(x))
    @pyjltry Cint(haskey(x, v)) Cint(-1)
end

pyjldict_get(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    ism1(PyArg_CheckNumArgsBetween("get", args, 1, 2)) && return PyNULL
    ko = PyTuple_GetItem(args, 0)
    vo = PyTuple_Size(args) < 2 ? Py_None() : PyTuple_GetItem(args, 1)
    r = PyObject_TryConvert(ko, keytype(x))
    r == -1 && return PyNULL
    r == 0 && (Py_IncRef(vo); return vo)
    k = takeresult(keytype(x))
    @pyjltry begin
        if haskey(x, k)
            PyObject_From(x[k])
        else
            Py_IncRef(vo)
            vo
        end
    end PyNULL
end

pyjldict_setdefault(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    ism1(PyArg_CheckNumArgsBetween("get", args, 1, 2)) && return PyNULL
    ko = PyTuple_GetItem(args, 0)
    vo = PyTuple_Size(args) < 2 ? Py_None() : PyTuple_GetItem(args, 1)
    r = PyObject_TryConvert(ko, keytype(x))
    r == -1 && return PyNULL
    r == 0 && (Py_IncRef(vo); return vo)
    k = takeresult(keytype(x))
    @pyjltry begin
        if haskey(x, k)
            PyObject_From(x[k])
        else
            r = PyObject_Convert(vo, valtype(x))
            r == -1 && return PyNULL
            x[k] = takeresult(valtype(x))
            Py_IncRef(vo)
            vo
        end
    end PyNULL
end

pyjldict_clear(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    @pyjltry begin
        empty!(x)
        PyNone_New()
    end PyNULL
end

pyjldict_popitem(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    @pyjltry begin
        if isempty(x)
            PyErr_SetString(PyExc_KeyError(), "pop empty dictionary")
            PyNULL
        else
            k, v = pop!(x)
            PyTuple_From((k, v))
        end
    end PyNULL
end

pyjldict_pop(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractDict
    ism1(PyArg_CheckNumArgsBetween("pop", args, 1, 2)) && return PyNULL
    ko = PyTuple_GetItem(args, 0)
    vo = PyTuple_Size(args) == 2 ? PyTuple_GetItem(args, 1) : PyNULL
    r = PyObject_TryConvert(PyTuple_GetItem(args, 0), keytype(x))
    r == -1 && return PyNULL
    r == 0 &&
        (isnull(vo) ? (PyErr_SetObject(PyExc_KeyError(), ko)) : (Py_IncRef(vo); vo))
    k = takeresult(keytype(x))
    @pyjltry begin
        if haskey(x, k)
            PyObject_From(pop!(x, k))
        elseif isnull(vo)
            PyErr_SetObject(PyExc_KeyError(), ko)
            PyNULL
        else
            Py_IncRef(vo)
            vo
        end
    end PyNULL
end

struct DictPairSet{K,V,T<:AbstractDict{K,V}} <: AbstractSet{Tuple{K,V}}
    dict::T
end
Base.length(x::DictPairSet) = length(x.dict)
Base.iterate(x::DictPairSet) =
    (r = iterate(x.dict); r === nothing ? nothing : (Tuple(r[1]), r[2]))
Base.iterate(x::DictPairSet, st) =
    (r = iterate(x.dict, st); r === nothing ? nothing : (Tuple(r[1]), r[2]))
Base.in(v, x::DictPairSet) = v in x.dict

const PyJuliaDictValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.DictValue"),
        base = base,
        iter = @cfunctionOO(pyjldict_iter),
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "keys"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjldict_keys),
            ),
            PyMethodDef(
                name = cacheptr!(c, "values"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjldict_values),
            ),
            PyMethodDef(
                name = cacheptr!(c, "items"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjldict_items),
            ),
            PyMethodDef(
                name = cacheptr!(c, "get"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOO(pyjldict_get),
            ),
            PyMethodDef(
                name = cacheptr!(c, "clear"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjldict_clear),
            ),
            PyMethodDef(
                name = cacheptr!(c, "pop"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOO(pyjldict_pop),
            ),
            PyMethodDef(
                name = cacheptr!(c, "popitem"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjldict_popitem),
            ),
            # PyMethodDef(
            #     name = cacheptr!(c, "update"),
            #     flags = Py_METH_VARARGS|Py_METH_KEYWORDS,
            #     meth = @cfunctionOOOO(pyjldict_update),
            # ),
            PyMethodDef(
                name = cacheptr!(c, "setdefault"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOO(pyjldict_setdefault),
            ),
            PyMethodDef(),
        ]),
        as_sequence = cacheptr!(c, fill(PySequenceMethods(
            contains = @cfunctionIOO(pyjldict_contains),
        ))),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyMutableMappingABC_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaDictValue_New(x::AbstractDict) = PyJuliaValue_New(PyJuliaDictValue_Type(), x)
PyJuliaValue_From(x::AbstractDict) = PyJuliaDictValue_New(x)
