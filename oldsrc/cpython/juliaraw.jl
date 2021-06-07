pyjlraw_repr(xo::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)
        s = "<jl $(repr(x))>"
        PyUnicode_From(s)
    end PyNULL

pyjlraw_str(xo::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)
        s = string(x)
        PyUnicode_From(s)
    end PyNULL

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))
pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

pyjlraw_getattro(xo::PyPtr, ko::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericGetAttr(xo, ko)
    if isnull(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    # Convert attribute to a string
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return PyNULL
    # If has double leading and trailing underscore, do not allow
    if length(k) > 4 && startswith(k, "__") && endswith(k, "__")
        PyErr_SetString(PyExc_AttributeError(), "'$(PyType_Name(Py_Type(xo)))' object has no attribute '$k'")
        return PyNULL
    end
    # Look up a property on the Julia object
    k = pyjl_attr_py2jl(k)
    @pyjltry PyJuliaRawValue_New(getproperty(x, Symbol(k))) PyNULL
end

pyjlraw_setattro(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericSetAttr(xo, ko, vo)
    if ism1(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    # Convert attribute to a string
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return Cint(-1)
    # If has double leading and trailing underscore, do not allow
    if length(k) > 4 && startswith(k, "__") && endswith(k, "__")
        PyErr_SetString(PyExc_AttributeError(), "'$(PyType_Name(Py_Type(xo)))' object has no attribute '$k'")
        return Cint(-1)
    end
    # Look up a property on the Julia object
    k = pyjl_attr_py2jl(k)
    ism1(PyObject_Convert(vo, Any)) && return Cint(-1)
    v = takeresult(Any)
    @pyjltry (setproperty!(x, Symbol(k), v); Cint(0)) Cint(-1)
end

pyjlraw_dir(xo::PyPtr, _::PyPtr) = begin
    fo = PyObject_GetAttrString(PyJuliaBaseValue_Type(), "__dir__")
    isnull(fo) && return PyNULL
    ro = PyObject_CallNice(fo, PyObjectRef(xo))
    Py_DecRef(fo)
    isnull(ro) && return PyNULL
    x = PyJuliaValue_GetValue(xo)
    ks = @pyjltry collect(map(string, propertynames(x))) PyNULL OnErr=>Py_DecRef(ro)
    for k in ks
        ko = PyUnicode_From(pyjl_attr_jl2py(k))
        isnull(ko) && (Py_DecRef(ro); return PyNULL)
        err = PyList_Append(ro, ko)
        Py_DecRef(ko)
        ism1(err) && (Py_DecRef(ro); return PyNULL)
    end
    return ro
end

pyjlraw_call(fo::PyPtr, argso::PyPtr, kwargso::PyPtr) = begin
    f = PyJuliaValue_GetValue(fo)
    if isnull(argso)
        args = Vector{Any}()
    else
        ism1(PyObject_Convert(argso, Vector{Any})) && return PyNULL
        args = takeresult(Vector{Any})
    end
    if isnull(kwargso)
        kwargs = Dict{Symbol,Any}()
    else
        ism1(PyObject_Convert(kwargso, Dict{Symbol,Any})) && return PyNULL
        kwargs = takeresult(Dict{Symbol,Any})
    end
    @pyjltry PyJuliaRawValue_New(f(args...; kwargs...)) PyNULL
end

pyjlraw_length(xo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    @pyjltry Py_ssize_t(length(x)) Py_ssize_t(-1)
end

pyjlraw_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    if PyTuple_Check(ko)
        ism1(PyObject_Convert(ko, Tuple)) && return PyNULL
        k = takeresult(Tuple)
        @pyjltry PyJuliaRawValue_New(x[k...]) PyNULL
    else
        ism1(PyObject_Convert(ko, Any)) && return PyNULL
        k = takeresult(Any)
        @pyjltry PyJuliaRawValue_New(x[k]) PyNULL
    end
end

pyjlraw_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    ism1(PyObject_Convert(vo, Any)) && return Cint(-1)
    v = takeresult(Any)
    if PyTuple_Check(ko)
        ism1(PyObject_Convert(ko, Tuple)) && return Cint(-1)
        k = takeresult(Tuple)
        @pyjltry (x[k...] = v; Cint(0)) Cint(-1)
    else
        ism1(PyObject_Convert(ko, Any)) && return Cint(-1)
        k = takeresult(Any)
        @pyjltry (x[k] = v; Cint(0)) Cint(-1)
    end
end

pyjlraw_toany(xo::PyPtr, ::PyPtr) = PyJuliaValue_From(PyJuliaValue_GetValue(xo))

const PyJuliaRawValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaBaseValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.RawValue"),
        base = base,
        repr = @cfunctionOO(pyjlraw_repr),
        str = @cfunctionOO(pyjlraw_str),
        getattro = @cfunctionOOO(pyjlraw_getattro),
        setattro = @cfunctionIOOO(pyjlraw_setattro),
        call = @cfunctionOOOO(pyjlraw_call),
        as_mapping = cacheptr!(c, fill(PyMappingMethods(
            length = @cfunctionZO(pyjlraw_length),
            subscript = @cfunctionOOO(pyjlraw_getitem),
            ass_subscript = @cfunctionIOOO(pyjlraw_setitem),
        ))),
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "__dir__"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlraw_dir),
            ),
            PyMethodDef(
                name = cacheptr!(c, "_jl_any"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlraw_toany),
            ),
            PyMethodDef(),
        ])
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaRawValue_New(x) = PyJuliaValue_New(PyJuliaRawValue_Type(), x)
