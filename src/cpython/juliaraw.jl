const PyJuliaRawValue_Type__ref = Ref(PyPtr())
PyJuliaRawValue_Type() = begin
    ptr = PyJuliaRawValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaBaseValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.RawValue",
            base = base,
            repr = pyjlraw_repr,
            str = pyjlraw_str,
            getattro = pyjlraw_getattro,
            setattro = pyjlraw_setattro,
            call = pyjlraw_call,
            as_mapping = (
                length = pyjlraw_length,
                subscript = pyjlraw_getitem,
                ass_subscript = pyjlraw_setitem,
            ),
            methods = [
                (name="__dir__", flags=Py_METH_NOARGS, meth=pyjlraw_dir),
                (name="__jl_any", flags=Py_METH_NOARGS, meth=pyjlraw_toany),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaRawValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaRawValue_New(x) = PyJuliaValue_New(PyJuliaRawValue_Type(), x)

pyjlraw_repr(xo::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)
    s = "<jl $(repr(x))>"
    PyUnicode_From(s)
catch err
    PyErr_SetJuliaError(err)
    return PyPtr()
end

pyjlraw_str(xo::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)
    s = string(x)
    PyUnicode_From(s)
catch err
    PyErr_SetJuliaError(err)
    return PyPtr()
end

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x)-1)))
pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

pyjlraw_getattro(xo::PyPtr, ko::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericGetAttr(xo, ko)
    if isnull(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    # Now try to get the corresponding property
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return PyPtr()
    k = pyjl_attr_py2jl(k)
    try
        v = getproperty(x, Symbol(k))
        PyJuliaRawValue_New(v)
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlraw_setattro(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericSetAttr(xo, ko, vo)
    if ism1(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    # Now try to set the corresponding property
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return Cint(-1)
    k = pyjl_attr_py2jl(k)
    ism1(PyObject_Convert(vo, Any)) && return Cint(-1)
    v = takeresult(Any)
    try
        setproperty!(x, Symbol(k), v)
        Cint(0)
    catch err
        PyErr_SetJuliaError(err)
        Cint(-1)
    end
end

pyjlraw_dir(xo::PyPtr, _::PyPtr) = begin
    fo = PyObject_GetAttrString(PyJuliaBaseValue_Type(), "__dir__")
    isnull(fo) && return PyPtr()
    ro = PyObject_CallNice(fo, PyObjectRef(xo))
    Py_DecRef(fo)
    isnull(ro) && return PyPtr()
    x = PyJuliaValue_GetValue(xo)
    ks = try
        collect(map(string, propertynames(x)))
    catch err
        Py_DecRef(ro)
        PyErr_SetJuliaError(err)
        return PyPtr()
    end
    for k in ks
        ko = PyUnicode_From(pyjl_attr_jl2py(k))
        isnull(ko) && (Py_DecRef(ro); return PyPtr())
        err = PyList_Append(ro, ko)
        Py_DecRef(ko)
        ism1(err) && (Py_DecRef(ro); return PyPtr())
    end
    return ro
end

pyjlraw_call(fo::PyPtr, argso::PyPtr, kwargso::PyPtr) = begin
    f = PyJuliaValue_GetValue(fo)
    if isnull(argso)
        args = Vector{Any}()
    else
        ism1(PyObject_Convert(argso, Vector{Any})) && return PyPtr()
        args = takeresult(Vector{Any})
    end
    if isnull(kwargso)
        kwargs = Dict{Symbol, Any}()
    else
        ism1(PyObject_Convert(kwargso, Dict{Symbol, Any})) && return PyPtr()
        kwargs = takeresult(Dict{Symbol, Any})
    end
    try
        x = f(args...; kwargs...)
        PyJuliaRawValue_New(x)
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlraw_length(xo::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)
    Py_ssize_t(length(x))
catch err
    PyErr_SetJuliaError(err)
    Py_ssize_t(-1)
end

pyjlraw_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    if PyTuple_Check(ko)
        ism1(PyObject_Convert(ko, Tuple)) && return PyPtr()
        k = takeresult(Tuple)
        try
            PyJuliaRawValue_New(x[k...])
        catch err
            PyErr_SetJuliaError(err)
            PyPtr()
        end
    else
        ism1(PyObject_Convert(ko, Any)) && return PyPtr()
        k = takeresult(Any)
        try
            PyJuliaRawValue_New(x[k])
        catch err
            PyErr_SetJuliaError(err)
            PyPtr()
        end
    end
end

pyjlraw_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    ism1(PyObject_Convert(vo, Any)) && return PyPtr()
    v = takeresult(Any)
    if PyTuple_Check(ko)
        ism1(PyObject_Convert(ko, Tuple)) && return PyPtr()
        k = takeresult(Tuple)
        try
            x[k...] = v
            Cint(0)
        catch err
            PyErr_SetJuliaError(err)
            Cint(-1)
        end
    else
        ism1(PyObject_Convert(ko, Any)) && return PyPtr()
        k = takeresult(Any)
        try
            x[k] = v
            Cint(0)
        catch err
            PyErr_SetJuliaError(err)
            Cint(-1)
        end
    end
end

pyjlraw_toany(xo::PyPtr, ::PyPtr) = PyJuliaValue_From(PyJuliaValue_GetValue(xo))
