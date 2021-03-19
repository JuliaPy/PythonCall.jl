pyjltype_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::Type
    if PyTuple_Check(ko)
        r = PyObject_Convert(ko, Tuple)
        ism1(r) && return PyNULL
        k = takeresult(Tuple)
        try
            PyObject_From(x{k...})
        catch err
            PyErr_SetJuliaError(err)
            PyNULL
        end
    else
        r = PyObject_Convert(ko, Any)
        ism1(r) && return PyNULL
        k = takeresult(Any)
        try
            PyObject_From(x{k})
        catch err
            PyErr_SetJuliaError(err)
            PyNULL
        end
    end
end

pyjltype_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    PyErr_SetString(PyExc_TypeError(), "Not supported.")
    Cint(-1)
end

PyJuliaTypeValue_New(x::Type) = PyJuliaValue_New(PyJuliaTypeValue_Type(), x)
PyJuliaValue_From(x::Type) = PyJuliaTypeValue_New(x)

const PyJuliaTypeValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.TypeValue"),
        base = base,
        as_mapping = cacheptr!(c, fill(PyMappingMethods(
            subscript = @cfunctionOOO(pyjltype_getitem),
            ass_subscript = @cfunctionIOOO(pyjltype_setitem),
        ))),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end
