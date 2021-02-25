const PyJuliaModuleValue_Type__ref = Ref(PyNULL)
PyJuliaModuleValue_Type() = begin
    ptr = PyJuliaModuleValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyNULL
        t = fill(
            PyType_Create(
                c,
                name = "juliaaa.ModuleValue",
                base = base,
                methods = [
                    (name = "seval", flags = Py_METH_VARARGS, meth = pyjlmodule_seval),
                ],
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyNULL
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaModuleValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaModuleValue_New(x::Module) = PyJuliaValue_New(PyJuliaModuleValue_Type(), x)
PyJuliaValue_From(x::Module) = PyJuliaModuleValue_New(x)

pyjlmodule_seval(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsBetween("seval", args, 1, 2)) && return PyNULL
    if PyTuple_Size(args) == 1
        m = PyJuliaValue_GetValue(xo)::Module
        ism1(PyArg_GetArg(String, "seval", args, 0)) && return PyNULL
        s = takeresult(String)
    else
        ism1(PyArg_GetArg(Module, "seval", args, 0)) && return PyNULL
        m = takeresult(Module)
        ism1(PyArg_GetArg(String, "seval", args, 1)) && return PyNULL
        s = takeresult(String)
    end
    try
        PyObject_From(Base.eval(m, Meta.parse(s)))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
end
