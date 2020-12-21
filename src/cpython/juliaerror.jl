const PyExc_JuliaError__ref = Ref(PyPtr())
PyExc_JuliaError() = begin
    ptr = PyExc_JuliaError__ref[]
    if isnull(ptr)
        c = []
        t = fill(PyType_Create(c,
            name = "julia.Error",
            base = PyExc_BaseException(),
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyExc_JuliaError__ref[] = ptr
    end
    ptr
end

PyErr_SetJuliaError(err, bt=nothing) = begin
    t = PyExc_JuliaError()
    if isnull(t)
        PyErr_Clear()
        PyErr_SetString(PyExc_Exception(), "Julia error: $err (an error occurred while setting this error)")
    else
        if bt === nothing
            bt = catch_backtrace()
        end
        v = PyJuliaBaseValue_New((err, bt))
        if isnull(v)
            PyErr_Clear()
            PyErr_SetString(t, string(err))
        else
            PyErr_SetObject(t, v)
            Py_DecRef(v)
        end
    end
end
