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
    @pyjltry PyObject_From(Base.eval(m, Meta.parse(s))) PyNULL
end

const PyJuliaModuleValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.ModuleValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "seval"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlmodule_seval),
            ),
            PyMethodDef(),
        ]),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaModuleValue_New(x::Module) = PyJuliaValue_New(PyJuliaModuleValue_Type(), x)
PyJuliaValue_From(x::Module) = PyJuliaModuleValue_New(x)
