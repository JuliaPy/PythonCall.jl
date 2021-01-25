const PyJuliaTypeValue_Type__ref = Ref(PyPtr())
PyJuliaTypeValue_Type() = begin
    ptr = PyJuliaTypeValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(
            PyType_Create(
                c,
                name = "julia.TypeValue",
                base = base,
                as_mapping = (
                    subscript = pyjltype_getitem,
                    ass_subscript = pyjltype_setitem,
                ),
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaTypeValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaTypeValue_New(x::Type) = PyJuliaValue_New(PyJuliaTypeValue_Type(), x)
PyJuliaValue_From(x::Type) = PyJuliaTypeValue_New(x)

pyjltype_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::Type
    if PyTuple_Check(ko)
        r = PyObject_Convert(ko, Tuple)
        ism1(r) && return PyPtr()
        k = takeresult(Tuple)
        try
            PyObject_From(x{k...})
        catch err
            PyErr_SetJuliaError(err)
            PyPtr()
        end
    else
        r = PyObject_Convert(ko, Any)
        ism1(r) && return PyPtr()
        k = takeresult(Any)
        try
            PyObject_From(x{k})
        catch err
            PyErr_SetJuliaError(err)
            PyPtr()
        end
    end
end

pyjltype_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    PyErr_SetString(PyExc_TypeError(), "Not supported.")
    PyErr()
end
