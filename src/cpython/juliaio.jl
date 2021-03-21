pyjlio_torawio(xo::PyPtr, ::PyPtr) = PyJuliaRawIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_tobufferedio(xo::PyPtr, ::PyPtr) =
    PyJuliaBufferedIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_totextio(xo::PyPtr, ::PyPtr) = PyJuliaTextIOValue_New(PyJuliaValue_GetValue(xo)::IO)
pyjlio_identity(xo::PyPtr, ::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_iter(xo::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_next(xo::PyPtr) = begin
    rl = PyObject_GetAttrString(xo, "readline")
    isnull(rl) && return PyNULL
    line = PyObject_CallNice(rl)
    Py_DecRef(rl)
    isnull(line) && return PyNULL
    if PyObject_Length(line) > 0
        line
    else
        Py_DecRef(line)
        PyNULL
    end
end

pyjlio_enter(xo::PyPtr, ::PyPtr) = (Py_IncRef(xo); xo)

pyjlio_exit(xo::PyPtr, args::PyPtr) = begin
    cl = PyObject_GetAttrString(xo, "close")
    isnull(cl) && return PyNULL
    v = PyObject_CallNice(cl)
    Py_DecRef(cl)
    Py_DecRef(v)
    isnull(v) && return PyNULL
    PyNone_New()
end

pyjlio_close(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry (close(x); PyNone_New()) PyNULL (MethodError, close)=>ValueError
end

pyjlio_closed(xo::PyPtr, ::Ptr{Cvoid}) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(!isopen(x)) PyNULL (MethodError, isopen)=>ValueError
end

pyjlio_fileno(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(fd(x)) PyNULL (MethodError, fd)=>ValueError
end

pyjlio_flush(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry (flush(x); PyNone_New()) PyNULL (MethodError, flush)=>ValueError
end

pyjlio_isatty(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(x isa Base.TTY) PyNULL
end

pyjlio_readable(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(isreadable(x)) PyNULL (MethodError, isreadable)=>ValueError
end

pyjlio_writable(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(iswritable(x)) PyNULL (MethodError, iswritable)=>ValueError
end

pyjlio_tell(xo::PyPtr, ::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry PyObject_From(position(x)) PyNULL (MethodError, position)=>ValueError
end

pyjlio_seek(xo::PyPtr, args::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    ism1(PyArg_CheckNumArgsBetween("seek", args, 1, 2)) && return PyNULL
    ism1(PyArg_GetArg(Int, "seek", args, 0)) && return PyNULL
    n = takeresult(Int)
    ism1(PyArg_GetArg(Int, "seek", args, 1, 0)) && return PyNULL
    w = takeresult(Int)
    @pyjltry begin
        if w == 0
            seek(x, n)
        elseif w == 1
            seek(x, position(x) + n)
        elseif w == 2
            seekend(x)
            seek(x, position(x) + n)
        else
            PyErr_SetString(PyExc_ValueError(), "'whence' argument must be 0, 1 or 2 (got $w)")
            return PyNULL
        end
        PyObject_From(position(x))
    end PyNULL (MethodError, seek, position, seekend)=>ValueError
end

pyjlio_writelines(xo::PyPtr, lines::PyPtr) = begin
    wr = PyObject_GetAttrString(xo, "write")
    isnull(wr) && return PyNULL
    r = PyIterable_Map(lines) do line
        r = PyObject_CallNice(wr, PyObjectRef(line))
        isnull(r) ? -1 : 1
    end
    r == -1 ? PyNULL : PyNone_New()
end

pyjlio_readlines(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("readlines", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Int, "readlines", args, 0, -1)) && return PyNULL
    limit = takeresult(Int)
    rl = PyObject_GetAttrString(xo, "readline")
    isnull(rl) && return PyNULL
    lines = PyList_New(0)
    isnull(lines) && (Py_DecRef(rl); return PyNULL)
    nread = 0
    while limit < 0 || nread < limit
        line = PyObject_CallNice(rl)
        isnull(line) && (Py_DecRef(lines); Py_DecRef(rl); return PyNULL)
        len = PyObject_Length(line)
        ism1(len) && (Py_DecRef(line); Py_DecRef(lines); Py_DecRef(rl); return PyNULL)
        len == 0 && (Py_DecRef(line); break)
        err = PyList_Append(lines, line)
        Py_DecRef(line)
        ism1(err) && (Py_DecRef(lines); Py_DecRef(rl); return PyNULL)
        nread += len
    end
    Py_DecRef(rl)
    lines
end

pyjlio_truncate(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("truncate", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Union{Int,Nothing}, "truncate", args, 0, nothing)) &&
        return PyNULL
    n = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry begin
        m = n === nothing ? position(x) : n
        truncate(x, m)
        PyObject_From(m)
    end PyNULL (MethodError, truncate, position)=>ValueError
end

pyjlio_seekable(xo::PyPtr, ::PyPtr) = begin
    T = typeof(PyJuliaValue_GetValue(xo)::IO)
    PyObject_From(hasmethod(seek, Tuple{T,Int}) && hasmethod(position, Tuple{T}))
end

pyjlio_encoding(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("UTF-8")
pyjlio_errors(::PyPtr, ::Ptr{Cvoid}) = PyUnicode_From("strict")
pyjlio_detach(o::PyPtr, ::PyPtr) = (
    PyErr_SetString(PyExc_ValueError(), "cannot detach '$(PyType_Name(Py_Type(o)))'"); PyNULL
)

pyjlio_write_bytes(xo::PyPtr, bo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    b = Ref(Py_buffer())
    ism1(PyObject_GetBuffer(bo, b, PyBUF_SIMPLE)) && return PyNULL
    s = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(b[].buf), b[].len)
    @pyjltry begin
        write(x, s)
        PyObject_From(length(s))
    end PyNULL (MethodError, write)=>ValueError Finally=>PyBuffer_Release(b)
end

pyjlio_write_str(xo::PyPtr, so::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::IO
    ism1(PyObject_Convert(so, String)) && return PyNULL
    s = takeresult(String)
    sep = ""
    havesep = false
    @pyjltry begin
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
                    isnull(m) && return PyNULL
                    sepo = PyObject_GetAttrString(m, "linesep")
                    isnull(sepo) && return PyNULL
                    r = PyObject_TryConvert(sepo, String)
                    Py_DecRef(sepo)
                    r == -1 && return PyNULL
                    r == 0 && (
                        PyErr_SetString(PyExc_TypeError(), "os.linesep must be a str"); return PyNULL
                    )
                    sep = takeresult(String)
                    havesep = true
                end
                write(x, sep)
                i = nextind(s, j)
            end
        end
        PyObject_From(length(s))
    end PyNULL (MethodError, write)=>ValueError
end

pyjlio_readinto(xo::PyPtr, bo::PyPtr) = begin
    b = Ref(Py_buffer())
    ism1(PyObject_GetBuffer(bo, b, PyBUF_WRITABLE)) && return PyNULL
    x = PyJuliaValue_GetValue(xo)::IO
    a = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(b[].buf), b[].len)
    @pyjltry PyObject_From(readbytes!(x, a, length(a))) PyNULL (MethodError, readbytes!)=>ValueError Finally=>PyBuffer_Release(b)
end

readpybytes(io::IO, limit = nothing) = begin
    buf = UInt8[]
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, UInt8)
        push!(buf, c)
    end
    buf
end

pyjlio_read_bytes(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("read", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Union{Int,Nothing}, "read", args, 0, nothing)) && return PyNULL
    limit = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry begin
        PyBytes_From(readpybytes(x, (limit === nothing || limit < 0) ? nothing : limit))
    end PyNULL (MethodError, eof, read, peek)=>ValueError
end

readpystr(io::IO, limit = nothing) = begin
    buf = IOBuffer()
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, Char)
        if c == '\n'
            write(buf, '\n')
        elseif c == '\r'
            !eof(io) && peek(io) == 0x0A && read(io, UInt8)
            write(buf, '\n')
        else
            write(buf, c)
            len += 1
        end
    end
    String(take!(buf))
end

pyjlio_read_str(xo::PyPtr, args::PyPtr) = begin
    ism1(PyArg_CheckNumArgsLe("read", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Union{Int,Nothing}, "read", args, 0, nothing)) && return PyNULL
    limit = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry begin
        PyObject_From(readpystr(x, (limit === nothing || limit < 0) ? nothing : limit))
    end PyNULL (MethodError, eof, read, peek)=>ValueError
end

readpybyteline(io::IO, limit = nothing) = begin
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
    ism1(PyArg_CheckNumArgsLe("readline", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Union{Int,Nothing}, "readline", args, 0, nothing)) &&
        return PyNULL
    limit = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry begin
        PyBytes_From(readpybyteline(x, (limit === nothing || limit < 0) ? nothing : limit))
    end PyNULL (MethodError, eof, read, peek)=>ValueError
end

readpyline(io::IO, limit = nothing) = begin
    buf = IOBuffer()
    len = 0
    while (limit === nothing || len < limit) && !eof(io)
        c = read(io, Char)
        if c == '\n'
            write(buf, '\n')
            break
        elseif c == '\r'
            !eof(io) && peek(io) == 0x0A && read(io, UInt8)
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
    ism1(PyArg_CheckNumArgsLe("readline", args, 1)) && return PyNULL
    ism1(PyArg_GetArg(Union{Int,Nothing}, "readline", args, 0, nothing)) &&
        return PyNULL
    limit = takeresult(Union{Int,Nothing})
    x = PyJuliaValue_GetValue(xo)::IO
    @pyjltry begin
        PyObject_From(readpyline(x, (limit === nothing || limit < 0) ? nothing : limit))
    end PyNULL (MethodError, eof, read, peek)=>ValueError
end

const PyJuliaIOValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.IOValue"),
        base = base,
        iter = @cfunctionOO(pyjlio_iter),
        iternext = @cfunctionOO(pyjlio_next),
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "close"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_close),
            ),
            PyMethodDef(
                name = cacheptr!(c, "fileno"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_fileno),
            ),
            PyMethodDef(
                name = cacheptr!(c, "flush"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_flush),
            ),
            PyMethodDef(
                name = cacheptr!(c, "isatty"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_isatty),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readable"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_readable),
            ),
            PyMethodDef(
                name = cacheptr!(c, "writable"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_writable),
            ),
            PyMethodDef(
                name = cacheptr!(c, "tell"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_tell),
            ),
            PyMethodDef(
                name = cacheptr!(c, "seek"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_seek),
            ),
            PyMethodDef(
                name = cacheptr!(c, "writelines"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_writelines),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readlines"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_readlines),
            ),
            PyMethodDef(
                name = cacheptr!(c, "truncate"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_truncate),
            ),
            PyMethodDef(
                name = cacheptr!(c, "seekable"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_seekable),
            ),
            PyMethodDef(
                name = cacheptr!(c, "__enter__"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_enter),
            ),
            PyMethodDef(
                name = cacheptr!(c, "__exit__"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_exit),
            ),
            PyMethodDef(
                name = cacheptr!(c, "torawio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_torawio),
            ),
            PyMethodDef(
                name = cacheptr!(c, "tobufferedio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_tobufferedio),
            ),
            PyMethodDef(
                name = cacheptr!(c, "totextio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_totextio),
            ),
            PyMethodDef(),
        ]),
        getset = cacheptr!(c, [
            PyGetSetDef(
                name = cacheptr!(c, "closed"),
                get = @cfunctionOOP(pyjlio_closed),
            ),
            PyGetSetDef(),
        ])
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyIOBase_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

const PyJuliaRawIOValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaIOValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.RawIOValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "write"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_write_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readall"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_read_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "read"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_read_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readline"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_readline_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readinto"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_readinto),
            ),
            PyMethodDef(
                name = cacheptr!(c, "torawio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_identity),
            ),
            PyMethodDef(),
        ]),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyRawIOBase_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

const PyJuliaBufferedIOValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaIOValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.BufferedIOValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "detach"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_detach),
            ),
            PyMethodDef(
                name = cacheptr!(c, "write"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_write_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "read"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_read_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "read1"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_read_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readline"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_readline_bytes),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readinto"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_readinto),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readinto1"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_readinto),
            ),
            PyMethodDef(
                name = cacheptr!(c, "tobufferedio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_identity),
            ),
            PyMethodDef(),
        ])
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyBufferedIOBase_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

const PyJuliaTextIOValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaIOValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.TextIOValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "detach"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_detach),
            ),
            PyMethodDef(
                name = cacheptr!(c, "write"),
                flags = Py_METH_O,
                meth = @cfunctionOOO(pyjlio_write_str),
            ),
            PyMethodDef(
                name = cacheptr!(c, "read"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_read_str),
            ),
            PyMethodDef(
                name = cacheptr!(c, "readline"),
                flags = Py_METH_VARARGS,
                meth = @cfunctionOOO(pyjlio_readline_str),
            ),
            PyMethodDef(
                name = cacheptr!(c, "totextio"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlio_identity),
            ),
            PyMethodDef(),
        ]),
        getset = cacheptr!(c, [
            PyGetSetDef(
                name = cacheptr!(c, "encoding"),
                get = @cfunctionOOP(pyjlio_encoding),
            ),
            PyGetSetDef(
                name = cacheptr!(c, "errors"),
                get = @cfunctionOOP(pyjlio_errors),
            ),
            PyGetSetDef(),
        ]),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyTextIOBase_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaIOValue_Type(), x)
PyJuliaRawIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaRawIOValue_Type(), x)
PyJuliaBufferedIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaBufferedIOValue_Type(), x)
PyJuliaTextIOValue_New(x::IO) = PyJuliaValue_New(PyJuliaTextIOValue_Type(), x)
PyJuliaValue_From(x::IO) = PyJuliaBufferedIOValue_New(x)
