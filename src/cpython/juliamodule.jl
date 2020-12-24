const PyJuliaModuleValue_Type__ref = Ref(PyPtr())
PyJuliaModuleValue_Type() = begin
    ptr = PyJuliaModuleValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.ModuleValue",
            base = base,
            methods = [
                (name="eval", flags=Py_METH_VARARGS, meth=pyjlmodule_eval),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaModuleValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaModuleValue_New(x::Module) = PyJuliaValue_New(PyJuliaModuleValue_Type(), x)
PyJuliaValue_From(x::Module) = PyJuliaModuleValue_New(x)

pyjlmodule_eval(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::Module
    ism1(PyArg_CheckNumArgsBetween("eval", args, 1, 2)) && return PyPtr()
    if PyTuple_Size(args) == 1
        m = PyJuliaValue_GetValue(xo)::Module
        exo = PyTuple_GetItem(args, 0)
    else
        ism1(PyArg_GetArg(Module, "eval", args, 0)) && return PyPtr()
        m = takeresult(Module)
        exo = PyTuple_GetItem(args, 1)
    end
    try
        if PyUnicode_Check(exo)
            c = PyUnicode_AsString(exo)
            isempty(c) && PyErr_IsSet() && return PyPtr()
            ex = Meta.parse(c)
        else
            ism1(PyObject_Convert(exo, Any)) && return PyPtr()
            ex = takeresult()
        end
        r = Base.eval(m, ex)
        PyObject_From(r)
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end
