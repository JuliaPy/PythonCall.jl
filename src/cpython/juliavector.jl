const PyJuliaVectorValue_Type__ref = Ref(PyPtr())
PyJuliaVectorValue_Type() = begin
    ptr = PyJuliaVectorValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaArrayValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.VectorValue",
            base = base,
            methods = [
                (name="resize", flags=Py_METH_O, meth=pyjlvector_resize),
                (name="sort", flags=Py_METH_NOARGS, meth=pyjlvector_sort),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyMutableSequenceABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaVectorValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaVectorValue_New(x::AbstractVector) = PyJuliaValue_New(PyJuliaVectorValue_Type(), x)
PyJuliaValue_From(x::AbstractVector) = PyJuliaVectorValue_New(x)

pyjlvector_resize(xo::PyPtr, arg::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(arg, Int)
    r == -1 && return PyPtr()
    r ==  0 && (PyErr_SetString(PyExc_TypeError(), "size must be an integer"); return PyPtr())
    resize!(x, takeresult(Int))
    PyNone_New()
catch err
    PyErr_SetJuliaError(err)
    PyPtr()
end

pyjlvector_sort(xo::PyPtr, ::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    sort!(x)
    PyNone_New()
catch err
    PyErr_SetJuliaError(err)
    PyPtr()
end
