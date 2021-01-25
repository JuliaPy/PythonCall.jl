const PyExc_JuliaError__ref = Ref(PyPtr())
PyExc_JuliaError() = begin
    ptr = PyExc_JuliaError__ref[]
    if isnull(ptr)
        c = []
        t = fill(
            PyType_Create(
                c,
                name = "julia.Error",
                base = PyExc_BaseException(),
                str = pyjlerr_str,
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyExc_JuliaError__ref[] = ptr
    end
    ptr
end

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
    isnull(args) && return PyPtr()
    r = PyObject_TryConvert(args, Tuple{Union{String,Tuple}})
    r == -1 && (Py_DecRef(args); return PyPtr())
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
