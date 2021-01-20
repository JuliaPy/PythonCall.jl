const PyJuliaIOValue_Type__ref = Ref(PyPtr())
PyJuliaIOValue_Type() = begin
    ptr = PyJuliaIOValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.IOValue",
            base = base,
            methods = [
                (name="close", flags=Py_METH_NOARGS, meth=pyjlio_close),
                (name="fileno", flags=Py_METH_NOARGS, meth=pyjlio_fileno),
                (name="flush", flags=Py_METH_NOARGS, meth=pyjlio_flush),
                (name="isatty", flags=Py_METH_NOARGS, meth=pyjlio_isatty),
                (name="readable", flags=Py_METH_NOARGS, meth=pyjlio_readable),
                (name="writable", flags=Py_METH_NOARGS, meth=pyjlio_writable),
                (name="tell", flags=Py_METH_NOARGS, meth=pyjlio_tell),
                (name="seek", flags=Py_METH_VARARGS, meth=pyjlio_seek),
                (name="writelines", flags=Py_METH_O, meth=pyjlio_writelines),
                (name="readlines", flags=Py_METH_VARARGS, meth=pyjlio_readlines),
                (name="truncate", flags=Py_METH_VARARGS, meth=pyjlio_truncate),
                (name="seekable", flags=Py_METH_NOARGS, meth=pyjlio_seekable),
            ],
            getset = [
                (name="closed", get=pyjlio_closed),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyIOBase_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaIOValue_Type__ref[] = ptr
    end
    ptr
end

const PyJuliaRawIOValue_Type__ref = Ref(PyPtr())
PyJuliaRawIOValue_Type() = begin
    ptr = PyJuliaRawIOValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaIOValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.RawIOValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyRawIOBase_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaRawIOValue_Type__ref[] = ptr
    end
    ptr
end

const PyJuliaBufferedIOValue_Type__ref = Ref(PyPtr())
PyJuliaBufferedIOValue_Type() = begin
    ptr = PyJuliaBufferedIOValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaIOValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.BufferedIOValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyBufferedIOBase_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaBufferedIOValue_Type__ref[] = ptr
    end
    ptr
end

const PyJuliaTextIOValue_Type__ref = Ref(PyPtr())
PyJuliaTextIOValue_Type() = begin
    ptr = PyJuliaTextIOValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaIOValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.TextIOValue",
            base = base,
            methods = [
                (name="detach", flags=Py_METH_NOARGS, meth=pyjltextio_detach),
                (name="write", flags=Py_METH_O, meth=pyjltextio_write),
                (name="read", flags=Py_METH_VARARGS, meth=pyjltextio_read),
                (name="readline", flags=Py_METH_VARARGS, meth=pyjltextio_readline),
            ],
            getset = [
                (name="encoding", get=pyjltextio_encoding),
                (name="errors", get=pyjltextio_errors),
            ],
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyTextIOBase_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaTextIOValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaIOValue_Type(), x)
PyJuliaRawIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaRawIOValue_Type(), x)
PyJuliaBufferedIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaBufferedIOValue_Type(), x)
PyJuliaTextIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaTextIOValue_Type(), x)
PyJuliaValue_From(x::IO) = PyJuliaBufferedIOValue_New(x)

# TODO:
#
# - IOBase:
#   __enter__() / __exit__()
#
# - RawIOBase:
#   read(size=-1)
#   readall()
#   readinto(b)
#   readline(size=-1)
#   write(b)
#
# - BufferedIOBase:
#   detach()
#   read(size=-1)
#   read1(size=-1)
#   readinto(b)
#   readinto1(b)
#   readline(size=-1)
#   write(b)

pyjlio_close(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        close(x)
        PyNone_New()
    catch err
        if err isa MethodError && err.f === close
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_closed(xo::PyPtr, ::Ptr{Cvoid}) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(!isopen(x))
    catch err
        if err isa MethodError && err.f === isopen
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_fileno(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(fd(x))
    catch err
        if err isa MethodError && err.f === fd
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_flush(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        flush(x)
        PyNone_New()
    catch err
        if err isa MethodError && err.f === flush
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_isatty(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(x isa Base.TTY)
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlio_readable(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(isreadable(x))
    catch err
        if err isa MethodError && err.f === isreadable
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_writable(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(iswritable(x))
    catch err
        if err isa MethodError && err.f === iswritable
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_tell(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(position(x))
    catch err
        if err isa MethodError && err.f === position
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_seek(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    ism1(PyArg_CheckNumArgsBetween("seek", args, 1, 2)) && return PyPtr()
    ism1(PyArg_GetArg(Int, "seek", args, 0)) && return PyPtr()
    n = takeresult(Int)
    ism1(PyArg_GetArg(Int, "seek", args, 1, 0)) && return PyPtr()
    w = takeresult(Int)
    try
        if w == 0
            seek(x, n)
        elseif w == 1
            seek(x, position(x) + n)
        elseif w == 2
            seekend(x)
            seek(x, position(x) + n)
        else
            PyErr_SetString(PyExc_ValueError(), "'whence' argument must be 0, 1 or 2 (got $w)")
            return PyPtr()
        end
        PyObject_From(position(x))
    catch err
        if err isa MethodError && (err.f === seek || err.f === position || err.f === seekend)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_writelines(xo::PyPtr, lines::PyPtr) = begin
    wr = PyObject_GetAttrString(xo, "write")
    isnull(wr) && return PyPtr()
    it = PyObject_GetIter(lines)
    isnull(it) && (Py_DecRef(wr); return PyPtr())
    while true
        line = PyIter_Next(it)
        if !isnull(line)
            ret = PyObject_CallNice(wr, PyObjectRef(line))
            Py_DecRef(line)
            isnull(ret) && (Py_DecRef(wr); Py_DecRef(it); return PyPtr())
            Py_DecRef(ret)
        else
            Py_DecRef(wr)
            Py_DecRef(it)
            return PyErr_IsSet() ? PyPtr() : PyNone_New()
        end
    end
end

pyjlio_readlines(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("readlines", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Int, "readlines", args, 0, -1)) && return PyPtr()
    limit = takeresult(Int)
    rl = PyObject_GetAttrString(xo, "readline")
    isnull(rl) && return PyPtr()
    lines = PyList_New(0)
    isnull(lines) && (Py_DecRef(rl); return PyPtr())
    nread = 0
    while limit < 0 || nread < limit
        line = PyObject_CallNice(rl)
        isnull(line) && (Py_DecRef(lines); Py_DecRef(rl); return PyPtr())
        len = PyObject_Length(line)
        ism1(len) && (Py_DecRef(line); Py_DecRef(lines); Py_DecRef(rl); return PyPtr())
        len == 0 && (Py_DecRef(line); break)
        err = PyList_Append(lines, line)
        Py_DecRef(line)
        ism1(err) && (Py_DecRef(lines); Py_DecRef(rl); return PyPtr())
        nread += len
    end
    Py_DecRef(rl)
    lines
end

pyjlio_truncate(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("truncate", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "truncate", args, 0, nothing)) && return PyPtr()
    n = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        m = n === nothing ? position(x) : n
        truncate(x, m)
        PyObject_From(m)
    catch err
        if err isa MethodError && (err.f === truncate || err.f === position)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

pyjlio_seekable(xo::PyPtr, ::PyPtr) = begin
    T = typeof(PyJuliaValue_GetValue(xo)::IO)
    PyObject_From(hasmethod(seek, Tuple{T,Int}) && hasmethod(position, Tuple{T}))
end

pyjltextio_encoding(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("UTF-8")
pyjltextio_errors(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("strict")
pyjltextio_detach(::PyPtr, ::PyPtr) = (PyErr_SetString(PyExc_ValueError(), "cannot detach"); PyPtr())

# TODO: translate '\n' to os.linesep
pyjltextio_write(xo::PyPtr, so::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    ism1(PyObject_Convert(so, String)) && return PyPtr()
    s = takeresult(String)
    try
        write(x, s)
        PyObject_From(length(s))
    catch err
        if err isa MethodError && err.f === write
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

readstringlimit(io::IO, limit::Int) = begin
    buf = IOBuffer()
    len = 0
    while !eof(io) && len < limit
        write(buf, read(io, Char))
        len += 1
    end
    String(take!(buf))
end

pyjltextio_read(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("read", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "read", args, 0, nothing)) && return PyPtr()
    limit = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        if limit === nothing || limit < 0
            PyObject_From(read(x, String))
        else
            PyObject_From(readstringlimit(x, limit))
        end
    catch err
        if err isa MethodError && (err.f === read || err.f === eof)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

readpyline(io::IO, limit=nothing) = begin
    buf = IOBuffer()
    len = 0
    while !eof(io) && (limit === nothing || len < limit)
        c = read(io, Char)
        if c == '\n'
            write(buf, '\n')
            break
        elseif c == '\r'
            !eof(io) && peek(io, Char)=='\n' && read(io, Char)
            write(buf, '\n')
            break
        else
            write(buf, c)
            len += 1
        end
    end
    String(take!(buf))
end

pyjltextio_readline(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("readline", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "readline", args, 0, nothing)) && return PyPtr()
    limit = takeresult(Union{Int, Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(readpyline(x, (limit===nothing || limit < 0) ? nothing : limit))
    catch err
        if err isa MethodError && (err.f === eof || err.f === read || err.f === peek)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end
