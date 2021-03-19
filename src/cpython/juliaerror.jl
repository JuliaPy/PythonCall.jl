PyErr_SetJuliaError(err, bt = nothing) = begin
    t = PyExc_JuliaError()
    if isnull(t)
        PyErr_Clear()
        PyErr_SetString(
            PyExc_Exception(),
            "Julia error: $err (an error occurred while setting this error)",
        )
    else
        if bt === nothing
            bt = catch_backtrace()
        end
        v = PyJuliaValue_From((err, bt))
        if isnull(v)
            PyErr_Clear()
            PyErr_SetString(t, string(err))
        else
            PyErr_SetObject(t, v)
            Py_DecRef(v)
        end
    end
end

PyErr_SetStringFromJuliaError(t, err) = begin
    io = IOBuffer()
    showerror(io, err)
    msg = String(take!(io))
    PyErr_SetString(t, "Julia: $msg")
end

pyjlerr_str(xo::PyPtr) = begin
    args = PyObject_GetAttrString(xo, "args")
    isnull(args) && return PyNULL
    r = PyObject_TryConvert(args, Tuple{Union{String,Tuple}})
    r == -1 && (Py_DecRef(args); return PyNULL)
    r == 0 && @goto fallback
    (x,) = takeresult(Tuple{Union{String,Tuple}})
    if x isa String
        Py_DecRef(args)
        return PyUnicode_From(x)
    elseif x isa Tuple{Exception,Any}
        Py_DecRef(args)
        io = IOBuffer()
        showerror(io, x[1])
        msg = String(take!(io))
        return PyUnicode_From(msg)
    end

    @label fallback
    so = PyObject_Str(args)
    Py_DecRef(args)
    return so
end

const PyExc_JuliaError = LazyPyObject() do
    c = []
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.Error"),
        base = PyExc_Exception(),
        str = @cfunctionOO(pyjlerr_str),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end
