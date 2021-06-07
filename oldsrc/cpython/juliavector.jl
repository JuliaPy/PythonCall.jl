pyjlvector_resize(xo::PyPtr, arg::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        r = PyObject_TryConvert(arg, Int)
        r == -1 && return PyNULL
        r == 0 &&
            (PyErr_SetString(PyExc_TypeError(), "size must be an integer"); return PyNULL)
        resize!(x, takeresult(Int))
        PyNone_New()
    end PyNULL

pyjlvector_sort(xo::PyPtr, args::PyPtr, kwargs::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        ism1(PyArg_CheckNumArgsEq("sort", args, 0)) && return PyNULL
        ism1(PyArg_GetArg(Bool, "sort", kwargs, "reverse", false)) && return PyNULL
        rev = takeresult(Bool)
        ism1(PyArg_GetArg(Any, "sort", kwargs, "key", nothing)) && return PyNULL
        by = takeresult()
        sort!(x, rev = rev, by = by === nothing ? identity : by)
        PyNone_New()
    end PyNULL

pyjlvector_reverse(xo::PyPtr, ::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        reverse!(x)
        PyNone_New()
    end PyNULL

pyjlvector_clear(xo::PyPtr, ::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        empty!(x)
        PyNone_New()
    end PyNULL

pyjlvector_reversed(xo::PyPtr, ::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        PyObject_From(reverse(x))
    end PyNULL

pyjlvector_insert(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    a = axes(x, 1)
    ism1(PyArg_CheckNumArgsEq("insert", args, 2)) && return PyNULL
    ism1(PyArg_GetArg(Int, "insert", args, 0)) && return PyNULL
    k = takeresult(Int)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    checkbounds(Bool, x, k′) || k′ == last(a)+1 || (
        PyErr_SetString(PyExc_IndexError(), "array index out of bounds"); return PyNULL
    )
    ism1(PyArg_GetArg(eltype(x), "insert", args, 1)) && return PyNULL
    v = takeresult(eltype(x))
    @pyjltry begin
        insert!(x, k′, v)
        PyNone_New()
    end PyNULL
end

pyjlvector_append(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 && (
        PyErr_SetString(
            PyExc_TypeError(),
            "array value must be a Julia '$(eltype(x))', not a '$(PyType_Name(Py_Type(vo)))'",
        );
        return PyNULL
    )
    v = takeresult(eltype(x))
    @pyjltry begin
        push!(x, v)
        PyNone_New()
    end PyNULL
end

pyjlvector_extend(xo::PyPtr, vso::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyIterable_Map(vso) do vo
        r = PyObject_Convert(vo, eltype(x))
        r == -1 && return -1
        v = takeresult(eltype(x))
        push!(x, v)
        return 1
    end
    r == -1 ? PyNULL : PyNone_New()
end

pyjlvector_pop(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    a = axes(x, 1)
    ism1(PyArg_CheckNumArgsLe("pop", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Int, "pop", args, 0, -1)) && return PyNULL
    k = takeresult(Int)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    checkbounds(Bool, x, k′) || (
        PyErr_SetString(PyExc_IndexError(), "array index out of bounds"); return PyNULL
    )
    @pyjltry begin
        if k′ == last(a)
            v = pop!(x)
        elseif k′ == first(a)
            v = popfirst!(x)
        else
            v = x[k′]
            deleteat!(x, k′)
        end
        PyObject_From(v)
    end PyNULL
end

pyjlvector_remove(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 &&
        (PyErr_SetString(PyExc_ValueError(), "value not in vector"); return PyNULL)
    v = takeresult(eltype(x))
    @pyjltry begin
        k = findfirst(x -> x == v, x)
        if k === nothing
            PyErr_SetString(PyExc_ValueError(), "value not in vector")
            PyNULL
        else
            deleteat!(x, k)
            PyNone_New()
        end
    end PyNULL
end

pyjlvector_index(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 &&
        (PyErr_SetString(PyExc_ValueError(), "value not in vector"); return PyNULL)
    v = takeresult(eltype(x))
    @pyjltry begin
        k = findfirst(x -> x == v, x)
        if k === nothing
            PyErr_SetString(PyExc_ValueError(), "value not in vector")
            PyNULL
        else
            PyObject_From(k - first(axes(x, 1)))
        end
    end PyNULL
end

pyjlvector_count(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 && return PyObject_From(0)
    v = takeresult(eltype(x))
    @pyjltry begin
        n = count(x -> x == v, x)
        PyObject_From(n)
    end PyNULL
end

const PyJuliaVectorValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaArrayValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.VectorValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "resize"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_resize),
            ),
            PyMethodDef(
                name = cacheptr!(c, "sort"),
                flags = Py_METH_VARARGS | Py_METH_KEYWORDS,
                meth = @cfunctionOOOO(pyjlvector_sort),
            ),
            PyMethodDef(
                name = cacheptr!(c, "reverse"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlvector_reverse),
            ),
            PyMethodDef(
                name = cacheptr!(c, "clear"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlvector_clear),
            ),
            PyMethodDef(
                name = cacheptr!(c, "__reversed__"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlvector_reversed),
            ),
            PyMethodDef(
                name = cacheptr!(c, "insert"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlvector_insert),
            ),
            PyMethodDef(
                name = cacheptr!(c, "append"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_append),
            ),
            PyMethodDef(
                name = cacheptr!(c, "extend"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_extend),
            ),
            PyMethodDef(
                name = cacheptr!(c, "pop"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlvector_pop),
            ),
            PyMethodDef(
                name = cacheptr!(c, "remove"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_remove),
            ),
            PyMethodDef(
                name = cacheptr!(c, "index"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_index),
            ),
            PyMethodDef(
                name = cacheptr!(c, "count"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlvector_count),
            ),
            PyMethodDef(),
        ])
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyMutableSequenceABC_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaVectorValue_New(x::AbstractVector) = PyJuliaValue_New(PyJuliaVectorValue_Type(), x)
PyJuliaValue_From(x::AbstractVector) = PyJuliaVectorValue_New(x)
