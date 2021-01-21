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
            iter = pyjlio_iter,
            iternext = pyjlio_next,
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
                (name="__enter__", flags=Py_METH_NOARGS, meth=pyjlio_enter),
                (name="__exit__", flags=Py_METH_VARARGS, meth=pyjlio_exit),
                # extras
                (name="torawio", flags=Py_METH_NOARGS, meth=pyjlio_torawio),
                (name="tobufferedio", flags=Py_METH_NOARGS, meth=pyjlio_tobufferedio),
                (name="totextio", flags=Py_METH_NOARGS, meth=pyjlio_totextio),
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
            methods = [
                (name="write", flags=Py_METH_O, meth=pyjlio_write_bytes),
                (name="readall", flags=Py_METH_VARARGS, meth=pyjlio_read_bytes),
                (name="read", flags=Py_METH_VARARGS, meth=pyjlio_read_bytes),
                (name="readline", flags=Py_METH_VARARGS, meth=pyjlio_readline_bytes),
                (name="readinto", flags=Py_METH_O, meth=pyjlio_readinto),
                # extras
                (name="torawio", flags=Py_METH_NOARGS, meth=pyjlio_identity),
            ],
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
            methods = [
                (name="detach", flags=Py_METH_NOARGS, meth=pyjlio_detach),
                (name="write", flags=Py_METH_O, meth=pyjlio_write_bytes),
                (name="read", flags=Py_METH_VARARGS, meth=pyjlio_read_bytes),
                (name="read1", flags=Py_METH_VARARGS, meth=pyjlio_read_bytes),
                (name="readline", flags=Py_METH_VARARGS, meth=pyjlio_readline_bytes),
                (name="readinto", flags=Py_METH_O, meth=pyjlio_readinto),
                (name="readinto1", flags=Py_METH_O, meth=pyjlio_readinto),
                # extras
                (name="tobufferedio", flags=Py_METH_NOARGS, meth=pyjlio_identity),
            ],
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
                (name="detach", flags=Py_METH_NOARGS, meth=pyjlio_detach),
                (name="write", flags=Py_METH_O, meth=pyjlio_write_str),
                (name="read", flags=Py_METH_VARARGS, meth=pyjlio_read_str),
                (name="readline", flags=Py_METH_VARARGS, meth=pyjlio_readline_str),
                # extras
                (name="totextio", flags=Py_METH_NOARGS, meth=pyjlio_identity),
            ],
            getset = [
                (name="encoding", get=pyjlio_encoding),
                (name="errors", get=pyjlio_errors),
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

pyjlio_torawio(xo::PyPtr, ::PyPtr) = PyJuliaRawIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_tobufferedio(xo::PyPtr, ::PyPtr) = PyJuliaBufferedIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_totextio(xo::PyPtr, ::PyPtr) = PyJuliaTextIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_identity(xo::PyPtr, ::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_iter(xo::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_next(xo::PyPtr) = begin
    rl = PyObject_GetAttrString(xo, "readline")
    isnull(rl) && return PyPtr()
    line = PyObject_CallNice(rl)
    Py_DecRef(rl)
    isnull(line) && return PyPtr()
    if PyObject_Length(line) > 0
        line
    else
        Py_DecRef(line)
        PyPtr()
    end
end

pyjlio_enter(xo::PyPtr, ::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_exit(xo::PyPtr, args::PyPtr) = begin
    cl = PyObject_GetAttrString(xo, "close")
    isnull(cl) && return PyPtr()
    v = PyObject_CallNice(cl)
    Py_DecRef(cl)
    Py_DecRef(v)
    isnull(v) && return PyPtr()
    PyNone_New()
end

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

pyjlio_encoding(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("UTF-8")
pyjlio_errors(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("strict")
pyjlio_detach(o::PyPtr, ::PyPtr) = (PyErr_SetString(PyExc_ValueError(), "cannot detach '$(PyType_Name(Py_Type(o)))'"); PyPtr())

pyjlio_write_bytes(xo::PyPtr, bo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    b = Ref(Py_buffer())
    ism1(PyObject_GetBuffer(bo, b, PyBUF_SIMPLE)) && return PyPtr()
    s = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(b[].buf), b[].len)
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
    finally
        PyBuffer_Release(b)
    end
end

pyjlio_write_str(xo::PyPtr, so::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    ism1(PyObject_Convert(so, String)) && return PyPtr()
    s = takeresult(String)
    sep = ""
    havesep = false
    try
        i = firstindex(s)
        while true
            j = findnext('\n', s, i)
            if j === nothing
                write(x, SubString(s, i))
                break
            else
                write(x, SubString(s, i, prevind(s, j)))
                if !havesep
                    m = Py_OSModule()
                    isnull(m) && return PyPtr()
                    sepo = PyObject_GetAttrString(m, "linesep")
                    isnull(sepo) && return PyPtr()
                    r = PyObject_TryConvert(sepo, String)
                    Py_DecRef(sepo)
                    r == -1 && return PyPtr()
                    r ==  0 && (PyErr_SetString(PyExc_TypeError(), "os.linesep must be a str"); return PyPtr())
                    sep = takeresult(String)
                    havesep = true
                end
                write(x, sep)
                i = nextind(s, j)
            end
        end
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

pyjlio_readinto(xo::PyPtr, bo::PyPtr) = begin
    b = Ref(Py_buffer())
    ism1(PyObject_GetBuffer(bo, b, PyBUF_WRITABLE)) && return PyPtr()
    x = PyJuliaValue_GetValue(xo)::IO
    a = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(b[].buf), b[].len)
    try
        PyObject_From(readbytes!(x, a, length(a)))
    catch err
        if err isa MethodError && err.f === readbytes!
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    finally
        PyBuffer_Release(b)
    end
end

readpybytes(io::IO, limit=nothing) = begin
    buf = UInt8[]
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, UInt8)
        push!(buf, c)
    end
    buf
end

pyjlio_read_bytes(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("read", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "read", args, 0, nothing)) && return PyPtr()
    limit = takeresult(Union{Int, Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyBytes_From(readpybytes(x, (limit===nothing || limit < 0) ? nothing : limit))
    catch err
        if err isa MethodError && (err.f === eof || err.f === read || err.f === peek)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

readpystr(io::IO, limit=nothing) = begin
    buf = IOBuffer()
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, Char)
        if c == '\n'
            write(buf, '\n')
        elseif c == '\r'
            !eof(io) && peek(io)==0x0A && read(io, UInt8)
            write(buf, '\n')
        else
            write(buf, c)
            len += 1
        end
    end
    String(take!(buf))
end

pyjlio_read_str(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("read", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "read", args, 0, nothing)) && return PyPtr()
    limit = takeresult(Union{Int, Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyObject_From(readpystr(x, (limit===nothing || limit < 0) ? nothing : limit))
    catch err
        if err isa MethodError && (err.f === eof || err.f === read || err.f === peek)
            PyErr_SetStringFromJuliaError(PyExc_ValueError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyPtr()
    end
end

readpybyteline(io::IO, limit=nothing) = begin
    buf = UInt8[]
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, UInt8)
        push!(buf, c)
        c == 0x0A && break
    end
    buf
end

pyjlio_readline_bytes(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("readline", args, 1)) && return PyPtr()
    ism1(PyArg_GetArg(Union{Int,Nothing}, "readline", args, 0, nothing)) && return PyPtr()
    limit = takeresult(Union{Int, Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    try
        PyBytes_From(readpybyteline(x, (limit===nothing || limit < 0) ? nothing : limit))
    catch err
        if err isa MethodError && (err.f === eof || err.f === read || err.f === peek)
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
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, Char)
        if c == '\n'
            write(buf, '\n')
            break
        elseif c == '\r'
            !eof(io) && peek(io)==0x0A && read(io, UInt8)
            write(buf, '\n')
            break
        else
            write(buf, c)
            len += 1
        end
    end
    String(take!(buf))
end

pyjlio_readline_str(xo::PyPtr, args::PyPtr) = begin
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
