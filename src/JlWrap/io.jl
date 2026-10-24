const pyjliobasetype = pynew()
const pyjlbinaryiotype = pynew()
const pyjltextiotype = pynew()

pyjlio_close(io::IO) = (close(io); Py(nothing))
pyjl_handle_error_type(::typeof(pyjlio_close), io, exc) =
    exc isa MethodError && exc.f === close ? pybuiltins.ValueError : PyNULL

pyjlio_closed(io::IO) = Py(!isopen(io))
pyjl_handle_error_type(::typeof(pyjlio_closed), io, exc) =
    exc isa MethodError && exc.f === isopen ? pybuiltins.ValueError : PyNULL

pyjlio_fileno(io::IO) = Py(fd(io))
pyjl_handle_error_type(::typeof(pyjlio_fileno), io, exc) =
    exc isa MethodError && exc.f === fd ? pybuiltins.ValueError : PyNULL

pyjlio_flush(io::IO) = (flush(io); Py(nothing))
pyjl_handle_error_type(::typeof(pyjlio_flush), io, exc) =
    exc isa MethodError && exc.f === flush ? pybuiltins.ValueError : PyNULL

pyjlio_isatty(io::IO) = Py(io isa Base.TTY)

pyjlio_readable(io::IO) = Py(isreadable(io))
pyjl_handle_error_type(::typeof(pyjlio_readable), io, exc) =
    exc isa MethodError && exc.f === isreadable ? pybuiltins.ValueError : PyNULL

function pyjlio_seek(io::IO, offset_::Py, whence_::Py)
    offset = pyconvertarg(Int, offset_, "offset")
    whence = pyconvertarg(Int, whence_, "whence")
    if whence == 0
        pos = offset
    elseif whence == 1
        pos = position(io) + offset
    elseif whence == 2
        seekend(io)
        pos = position(io) + offset
    else
        errset(pybuiltins.ValueError, "Argument 'whence' must be 0, 1 or 2")
        return PyNULL
    end
    seek(io, pos)
    Py(position(io))
end
pyjl_handle_error_type(::typeof(pyjlio_seek), io, exc) =
    exc isa MethodError && (exc.f === position || exc.f === seek || exc.f === seekend) ?
    pybuiltins.ValueError : PyNULL

pyjlio_tell(io::IO) = Py(position(io))
pyjl_handle_error_type(::typeof(pyjlio_tell), io, exc) =
    exc isa MethodError && exc.f === position ? pybuiltins.ValueError : PyNULL

function pyjlio_truncate(io::IO, size_::Py)
    size = pyconvertarg(Union{Int,Nothing}, size_, "size")
    if size === nothing
        size = position(io)
    end
    truncate(io, size)
    Py(size)
end
pyjl_handle_error_type(::typeof(pyjlio_truncate), io, exc) =
    exc isa MethodError && (exc.f === position || exc.f === truncate) ?
    pybuiltins.ValueError : PyNULL

pyjlio_writable(io::IO) = Py(iswritable(io))
pyjl_handle_error_type(::typeof(pyjlio_writable), io, exc) =
    exc isa MethodError && exc.f === iswritable ? pybuiltins.ValueError : PyNULL

function pyjlbinaryio_read(io::IO, size_::Py)
    size = pyconvertarg(Union{Int,Nothing}, size_, "size")
    if size === nothing || size < 0
        buf = read(io)
    else
        buf = read(io, size)
    end
    pybytes(buf)
end
pyjl_handle_error_type(::typeof(pyjlbinaryio_read), io, exc) =
    exc isa MethodError && exc.f === read ? pybuiltins.ValueError : PyNULL

function pyjlbinaryio_readline(io::IO, size_::Py)
    size = pyconvertarg(Union{Int,Nothing}, size_, "size")
    if size === nothing
        size = -1
    end
    buf = UInt8[]
    while !eof(io) && (size < 0 || length(buf) < size)
        c = read(io, UInt8)
        push!(buf, c)
        c == 0x0A && break
    end
    pybytes(buf)
end
pyjl_handle_error_type(::typeof(pyjlbinaryio_readline), io, exc) =
    exc isa MethodError && exc.f === read ? pybuiltins.ValueError : PyNULL

function pyjlbinaryio_readinto(io::IO, b::Py)
    m = pybuiltins.memoryview(b)
    c = m.c_contiguous
    if !pytruth(c)
        pydel!(c)
        errset(pybuiltins.ValueError, "input buffer is not contiguous")
        return PyNULL
    end
    pydel!(c)
    buf = unsafe_load(C.PyMemoryView_GET_BUFFER(m))
    if buf.readonly != 0
        pydel!(m)
        errset(pybuiltins.ValueError, "output buffer is read-only")
        return PyNULL
    end
    data = unsafe_wrap(Array, Ptr{UInt8}(buf.buf), buf.len)
    nb = readbytes!(io, data)
    pydel!(m)
    return Py(nb)
end
pyjl_handle_error_type(::typeof(pyjlbinaryio_readinto), io, exc) =
    exc isa MethodError && exc.f === readbytes! ? pybuiltins.ValueError : PyNULL

function pyjlbinaryio_write(io::IO, b::Py)
    m = pybuiltins.memoryview(b)
    c = m.c_contiguous
    if !pytruth(c)
        pydel!(c)
        errset(pybuiltins.ValueError, "input buffer is not contiguous")
        return PyNULL
    end
    pydel!(c)
    buf = unsafe_load(C.PyMemoryView_GET_BUFFER(m))
    data = unsafe_wrap(Array, Ptr{UInt8}(buf.buf), buf.len)
    write(io, data)
    pydel!(m)
    return Py(buf.len)
end
pyjl_handle_error_type(::typeof(pyjlbinaryio_write), io, exc) =
    exc isa MethodError && exc.f === write ? pybuiltins.ValueError : PyNULL

function pyjltextio_read(io::IO, size_::Py)
    size = pyconvertarg(Union{Int,Nothing}, size_, "size")
    if size === nothing
        size = -1
    end
    buf = IOBuffer()
    total = 0
    while !eof(io) && (size < 0 || total < size)
        c = read(io, Char)
        # translate "\n", "\r" and "\r\n" to "\n"
        if c == '\r'
            !eof(io) && peek(io) == 0x0A && read(io, UInt8)
            write(buf, '\n')
        else
            write(buf, c)
        end
        total += 1
    end
    Py(String(take!(buf)))
end
pyjl_handle_error_type(::typeof(pyjltextio_read), io, exc) =
    exc isa MethodError && exc.f === read ? pybuiltins.ValueError : PyNULL

function pyjltextio_readline(io::IO, size_::Py)
    size = pyconvertarg(Union{Int,Nothing}, size_, "size")
    if size === nothing
        size = -1
    end
    buf = IOBuffer()
    total = 0
    while !eof(io) && (size < 0 || total < size)
        c = read(io, Char)
        # translate "\n", "\r" and "\r\n" to "\n"
        if c == '\n'
            write(buf, c)
            total += 1
            break
        elseif c == '\r'
            !eof(io) && peek(io) == 0x0A && read(io, UInt8)
            write(buf, '\n')
            total += 1
            break
        else
            write(buf, c)
            total += 1
        end
    end
    Py(String(take!(buf)))
end
pyjl_handle_error_type(::typeof(pyjltextio_readline), io, exc) =
    exc isa MethodError && exc.f === read ? pybuiltins.ValueError : PyNULL

function pyjltextio_write(io::IO, s_::Py)
    if pyisstr(s_)
        s = pystr_asstring(s_)
        # get the line separator
        linesep_ = pyosmodule.linesep
        linesep = pystr_asstring(linesep_)
        pydel!(linesep_)
        # write the string
        # translating '\n' to os.linesep
        i = firstindex(s)
        iend = lastindex(s)
        while i ≤ iend
            j = findnext('\n', s, i)
            if j === nothing
                write(io, SubString(s, i))
                break
            else
                write(io, SubString(s, i, prevind(s, j)))
                write(io, linesep)
                i = nextind(s, j)
            end
        end
        # return number of characters written (not number of bytes)
        # TODO: is this the number of source characters, or the number of output characters?
        Py(length(s))
    else
        errset(
            pybuiltins.TypeError,
            "Argument 's' must be a 'str', got a '$(pytype(s_).__name__)'",
        )
        PyNULL
    end
end
pyjl_handle_error_type(::typeof(pyjltextio_write), io, exc) =
    exc isa MethodError && exc.f === write ? pybuiltins.ValueError : PyNULL

function init_io()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.io")
    methods = m._jl_methods
    methods["close"] = pyjl_methodnum(pyjlio_close)
    methods["closed"] = pyjl_methodnum(pyjlio_closed)
    methods["fileno"] = pyjl_methodnum(pyjlio_fileno)
    methods["flush"] = pyjl_methodnum(pyjlio_flush)
    methods["isatty"] = pyjl_methodnum(pyjlio_isatty)
    methods["readable"] = pyjl_methodnum(pyjlio_readable)
    methods["seek"] = pyjl_methodnum(pyjlio_seek)
    methods["tell"] = pyjl_methodnum(pyjlio_tell)
    methods["truncate"] = pyjl_methodnum(pyjlio_truncate)
    methods["writable"] = pyjl_methodnum(pyjlio_writable)
    methods["binary_read"] = pyjl_methodnum(pyjlbinaryio_read)
    methods["binary_readline"] = pyjl_methodnum(pyjlbinaryio_readline)
    methods["readinto"] = pyjl_methodnum(pyjlbinaryio_readinto)
    methods["binary_write"] = pyjl_methodnum(pyjlbinaryio_write)
    methods["text_read"] = pyjl_methodnum(pyjltextio_read)
    methods["text_readline"] = pyjl_methodnum(pyjltextio_readline)
    methods["text_write"] = pyjl_methodnum(pyjltextio_write)
    pycopy!(pyjliobasetype, jl.IOValueBase)
    pycopy!(pyjlbinaryiotype, jl.BinaryIOValue)
    pycopy!(pyjltextiotype, jl.TextIOValue)
end

pyiobase(v::IO) = pyjl(pyjliobasetype, v)

"""
    pybinaryio(io::IO)

Wrap `io` as a Python binary IO object.

This is the default behaviour of `Py(io)`.
"""
pybinaryio(v::IO) = pyjl(pyjlbinaryiotype, v)

"""
    pytextio(io::IO)

Wrap `io` as a Python text IO object.
"""
pytextio(v::IO) = pyjl(pyjltextiotype, v)

pyjltype(::IO) = pyjlbinaryiotype
