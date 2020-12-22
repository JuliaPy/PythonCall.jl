const PyJuliaArrayValue_Type__ref = Ref(PyPtr())
PyJuliaArrayValue_Type() = begin
    ptr = PyJuliaArrayValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.ArrayValue",
            base = base,
            as_mapping = (
                subscript = pyjlarray_getitem,
                ass_subscript = pyjlarray_setitem,
            ),
            getset = [
                (name="ndim", get=pyjlarray_ndim),
                (name="shape", get=pyjlarray_shape),
            ],
            methods = [
                (name="copy", flags=Py_METH_NOARGS, meth=pyjlarray_copy),
                (name="reshape", flags=Py_METH_O, meth=pyjlarray_reshape),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyCollectionABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaArrayValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaArrayValue_New(x::AbstractArray) = PyJuliaValue_New(PyJuliaArrayValue_Type(), x)
PyJuliaValue_From(x::AbstractArray) = PyJuliaArrayValue_New(x)

pyjl_getaxisindex(x::AbstractUnitRange{<:Integer}, ko::PyPtr) = begin
    if PySlice_Check(ko)
        ao, co, bo = PySimpleObject_GetValue(ko, Tuple{PyPtr, PyPtr, PyPtr})
        # start
        r = PyObject_TryConvert(ao, Union{Int, Nothing})
        r == -1 && return PYERR()
        r ==  0 && (PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR())
        a = takeresult(Union{Int, Nothing})
        # step
        r = PyObject_TryConvert(bo, Union{Int, Nothing})
        r == -1 && return PYERR()
        r ==  0 && (PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR())
        b = takeresult(Union{Int, Nothing})
        # stop
        r = PyObject_TryConvert(co, Union{Int, Nothing})
        r == -1 && return PYERR()
        r ==  0 && (PyErr_SetString(PyExc_TypeError(), "slice components must be integers"); return PYERR())
        c = takeresult(Union{Int, Nothing})
        # step defaults to 1
        b′ = b === nothing ? 1 : b
        if a === nothing && c === nothing
            # when neither is specified, start and stop default to the full range,
            # which is reversed when the step is negative
            if b′ > 0
                a′ = Int(first(x))
                c′ = Int(last(x))
            elseif b′ < 0
                a′ = Int(last(x))
                c′ = Int(first(x))
            else
                PyErr_SetString(PyExc_ValueError(), "step must be non-zero")
                return PYERR()
            end
        else
            # start defaults
            a′ = Int(a === nothing ? first(x) : a < 0 ? (last(x) + a + 1) : (first(x) + a))
            c′ = Int(c === nothing ? last(x) : c < 0 ? (last(x) + 1 + c - sign(b′)) : (first(x) + c - sign(b′)))
        end
        r = a′ : b′ : c′
        if !checkbounds(Bool, x, r)
            PyErr_SetString(PyExc_IndexError(), "array index out of bounds")
            return PYERR()
        end
        r
    else
        r = PyObject_TryConvert(ko, Int)
        ism1(r) && return PYERR()
        r == 0 && (PyErr_SetString(PyExc_TypeError(), "index must be slice or integer, got $(PyType_Name(Py_Type(ko)))"); return PYERR())
        k = takeresult(Int)
        k′ = k < 0 ? (last(x) + k + 1) : (first(x) + k)
        checkbounds(Bool, x, k′) || (PyErr_SetString(PyExc_IndexError(), "array index out of bounds"); return PYERR())
        k′
    end
end

pyjl_getarrayindices(x::AbstractArray, ko::PyPtr) = begin
    kos = PyTuple_Check(ko) ? [PyTuple_GetItem(ko, i-1) for i in 1:PyTuple_Size(ko)] : [ko]
    length(kos) == ndims(x) || (PyErr_SetString(PyExc_TypeError(), "expecting exactly $(ndims(x)) indices, got $(length(kos))"); return PYERR())
    ks = []
    for (i,ko) in enumerate(kos)
        k = pyjl_getaxisindex(axes(x, i), ko)
        k === PYERR() && return PYERR()
        push!(ks, k)
    end
    ks
end

pyjlarray_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    k = pyjl_getarrayindices(x, ko)
    k === PYERR() && return PyPtr()
    try
        if all(x -> x isa Int, k)
            PyObject_From(x[k...])
        else
            PyObject_From(view(x, k...))
        end
    catch err
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlarray_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    k = pyjl_getarrayindices(x, ko)
    k === PYERR() && return Cint(-1)
    try
        if isnull(vo)
            deleteat!(x, k...)
            Cint(0)
        else
            ism1(PyObject_Convert(vo, eltype(x))) && return PyPtr()
            v = takeresult(eltype(x))
            if all(x -> x isa Int, k)
                x[k...] = v
                Cint(0)
            else
                PyErr_SetString(PyExc_TypeError(), "multiple assignment not supported")
                Cint(-1)
            end
        end
    catch
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        elseif err isa MethodError && err.f === deleteat!
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Cint(-1)
    end
end

pyjlarray_ndim(xo::PyPtr, ::Ptr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    PyObject_From(ndims(x))
end

pyjlarray_shape(xo::PyPtr, ::Ptr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    PyObject_From(size(x))
end

pyjlarray_copy(xo::PyPtr, ::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    PyObject_From(copy(x))
catch err
    PyErr_SetJuliaError(err)
    PyPtr()
end

pyjlarray_reshape(xo::PyPtr, arg::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)::AbstractArray
    r = PyObject_TryConvert(arg, Union{Int, Tuple{Vararg{Int}}})
    r == -1 && return PyPtr()
    r ==  0 && (PyErr_SetString(PyExc_TypeError(), "shape must be an integer or tuple of integers"); return PyPtr())
    PyObject_From(reshape(x, takeresult()...))
catch err
    PyErr_SetJuliaError(err)
    PyPtr()
end
