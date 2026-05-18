const pyjliobasetype = pynew()
const pyjlbinaryiotype = pynew()
const pyjltextiotype = pynew()

pyjlio_close(io::IO) = (close(io); Py(nothing))
pyjl_handle_error_type(::typeof(pyjlio_close), io, exc) =
    exc isa MethodError && exc.f === close ? pybuiltins.ValueError : PyNULL

pyjlio_closed(io::IO) = Py(!isopen(io))
pyjl_handle_error_type(::typeof(pyjlio_closed), io, exc) =
    exc isa MethodError && exc.f === isopen ? pybuiltins.ValueError : PyNULL

pyjlio_fileno(io::IO) = Py(Base.cconvert(Cint, fd(io))::Cint)
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
    pybuiltins.exec(
        pybuiltins.compile(
            """
            $("\n"^(@__LINE__()-1))
            class JlIOBase(JlBase2):
                __slots__ = ()
                def __init__(self, value):
                    JlBase.__init__(self, value, Base.IO)
                def __hash__(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlany_hash)))
                def close(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_close)))
                @property
                def closed(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_closed)))
                def fileno(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_fileno)))
                def flush(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_flush)))
                def isatty(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_isatty)))
                def readable(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_readable)))
                def readlines(self, hint=-1):
                    lines = []
                    total = 0
                    while hint < 0 or total < hint:
                        line = self.readline()
                        if line:
                            lines.append(line)
                            total += len(line)
                        else:
                            break
                    return lines
                def seek(self, offset, whence=0):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_seek)), offset, whence)
                def seekable(self):
                    return True
                def tell(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_tell)))
                def truncate(self, size=None):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_truncate)), size)
                def writable(self):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlio_writable)))
                def writelines(self, lines):
                    for line in lines:
                        self.write(line)
                def __enter__(self):
                    return self
                def __exit__(self, t, v, b):
                    self.close()
                def __iter__(self):
                    return self
                def __next__(self):
                    line = self.readline()
                    if line:
                        return line
                    else:
                        raise StopIteration
            class JlBinaryIO(JlIOBase):
                __slots__ = ()
                def detach(self):
                    raise ValueError("Cannot detach '{}'.".format(type(self)))
                def read(self, size=-1):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlbinaryio_read)), size)
                def read1(self, size=-1):
                    return self.read(size)
                def readline(self, size=-1):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlbinaryio_readline)), size)
                def readinto(self, b):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlbinaryio_readinto)), b)
                def readinto1(self, b):
                    return self.readinto(b)
                def write(self, b):
                    return self._jl_callmethod($(pyjl_methodnum(pyjlbinaryio_write)), b)
            class JlTextIO(JlIOBase):
                __slots__ = ()
                @property
                def encoding(self):
                    return "UTF-8"
                @property
                def errors(self):
                    return "strict"
                def detach(self):
                    raise ValueError("Cannot detach '{}'.".format(type(self)))
                def read(self, size=-1):
                    return self._jl_callmethod($(pyjl_methodnum(pyjltextio_read)), size)
                def readline(self, size=-1):
                    return self._jl_callmethod($(pyjl_methodnum(pyjltextio_readline)), size)
                def write(self, s):
                    return self._jl_callmethod($(pyjl_methodnum(pyjltextio_write)), s)
            import io
            io.IOBase.register(JlIOBase)
            io.BufferedIOBase.register(JlBinaryIO)
            io.TextIOBase.register(JlTextIO)
            del io
            """,
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
    pycopy!(pyjliobasetype, jl.JlIOBase)
    pycopy!(pyjlbinaryiotype, jl.JlBinaryIO)
    pycopy!(pyjltextiotype, jl.JlTextIO)
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

Py(x::IO) = pybinaryio(x)
