const PyJuliaVectorValue_Type__ref = Ref(PyNULL)
PyJuliaVectorValue_Type() = begin
    ptr = PyJuliaVectorValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaArrayValue_Type()
        isnull(base) && return PyNULL
        t = fill(
            PyType_Create(
                c,
                name = "juliacall.VectorValue",
                base = base,
                methods = [
                    (name = "resize", flags = Py_METH_O, meth = pyjlvector_resize),
                    (
                        name = "sort",
                        flags = Py_METH_KEYWORDS | Py_METH_VARARGS,
                        meth = pyjlvector_sort,
                    ),
                    (name = "reverse", flags = Py_METH_NOARGS, meth = pyjlvector_reverse),
                    (name = "clear", flags = Py_METH_NOARGS, meth = pyjlvector_clear),
                    (
                        name = "__reversed__",
                        flags = Py_METH_NOARGS,
                        meth = pyjlvector_reversed,
                    ),
                    (name = "insert", flags = Py_METH_VARARGS, meth = pyjlvector_insert),
                    (name = "append", flags = Py_METH_O, meth = pyjlvector_append),
                    (name = "extend", flags = Py_METH_O, meth = pyjlvector_extend),
                    (name = "pop", flags = Py_METH_VARARGS, meth = pyjlvector_pop),
                    (name = "remove", flags = Py_METH_O, meth = pyjlvector_remove),
                    (name = "index", flags = Py_METH_O, meth = pyjlvector_index),
                    (name = "count", flags = Py_METH_O, meth = pyjlvector_count),
                ],
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyNULL
        abc = PyMutableSequenceABC_Type()
        isnull(abc) && return PyNULL
        ism1(PyABC_Register(ptr, abc)) && return PyNULL
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaVectorValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaVectorValue_New(x::AbstractVector) = PyJuliaValue_New(PyJuliaVectorValue_Type(), x)
PyJuliaValue_From(x::AbstractVector) = PyJuliaVectorValue_New(x)

pyjlvector_resize(xo::PyPtr, arg::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        r = PyObject_TryConvert(arg, Int)
        r == -1 && return PyNULL
        r == 0 &&
            (PyErr_SetString(PyExc_TypeError(), "size must be an integer"); return PyNULL)
        resize!(x, takeresult(Int))
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlvector_sort(xo::PyPtr, args::PyPtr, kwargs::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        ism1(PyArg_CheckNumArgsEq("sort", args, 0)) && return PyNULL
        ism1(PyArg_GetArg(Bool, "sort", kwargs, "reverse", false)) && return PyNULL
        rev = takeresult(Bool)
        ism1(PyArg_GetArg(Any, "sort", kwargs, "key", nothing)) && return PyNULL
        by = takeresult()
        sort!(x, rev = rev, by = by === nothing ? identity : by)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlvector_reverse(xo::PyPtr, ::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        reverse!(x)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlvector_clear(xo::PyPtr, ::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        empty!(x)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlvector_reversed(xo::PyPtr, ::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::AbstractVector
        PyObject_From(reverse(x))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlvector_insert(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    a = axes(x, 1)
    ism1(PyArg_CheckNumArgsEq("insert", args, 2)) && return PyNULL
    ism1(PyArg_GetArg(Int, "insert", args, 0)) && return PyNULL
    k = takeresult(Int)
    k′ = k < 0 ? (last(a) + 1 + k) : (first(a) + k)
    checkbounds(Bool, x, k′) || (
        PyErr_SetString(PyExc_IndexError(), "array index out of bounds"); return PyNULL
    )
    ism1(PyArg_GetArg(eltype(x), "insert", args, 1)) && return PyNULL
    v = takeresult(eltype(x))
    try
        insert!(x, k′, v)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
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
    try
        push!(x, v)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
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
    try
        if k′ == last(a)
            v = pop!(x)
        elseif k′ == first(a)
            v = popfirst!(x)
        else
            v = x[k′]
            deleteat!(x, k′)
        end
        PyObject_From(v)
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
end

pyjlvector_remove(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 &&
        (PyErr_SetString(PyExc_ValueError(), "value not in vector"); return PyNULL)
    v = takeresult(eltype(x))
    try
        k = findfirst(x -> x == v, x)
        if k === nothing
            PyErr_SetString(PyExc_ValueError(), "value not in vector")
            PyNULL
        else
            deleteat!(x, k)
            PyNone_New()
        end
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
end

pyjlvector_index(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 &&
        (PyErr_SetString(PyExc_ValueError(), "value not in vector"); return PyNULL)
    v = takeresult(eltype(x))
    try
        k = findfirst(x -> x == v, x)
        if k === nothing
            PyErr_SetString(PyExc_ValueError(), "value not in vector")
            PyNULL
        else
            PyObject_From(k - first(axes(x, 1)))
        end
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
end

pyjlvector_count(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractVector
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 && return PyObject_From(0)
    v = takeresult(eltype(x))
    try
        n = count(x -> x == v, x)
        PyObject_From(n)
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
end
