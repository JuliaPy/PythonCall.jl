"""
    PyIO(x; own=false, text=missing, buflen=4096)

Wrap the Python IO stream `x` as a Julia IO stream.

When this goes out of scope and is finalized, it is automatically flushed. If `own=true` then it is also closed.

If `text=false` then `x` must be a binary stream and arbitrary binary I/O is possible.
If `text=true` then `x` must be a text stream and only UTF-8 must be written (i.e. use `print` not `write`).
If `text` is not specified then it is chosen automatically.
If `x` is a text stream and you really need a binary stream, then often `PyIO(x.buffer)` will work.

For efficiency, reads and writes are buffered before being sent to `x`.
The size of the buffers is `buflen`.
The buffers are cleared using `flush`.
"""
mutable struct PyIO <: IO
    py::Py
    # true to close the file automatically
    own::Bool
    # true if `o` is text, false if binary
    text::Bool
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof::Bool
    # input buffer
    ibuflen::Int
    ibuf::Vector{UInt8}
    # output buffer
    obuflen::Int
    obuf::Vector{UInt8}

    PyIO(::Val{:new}, py::Py, own::Bool, text::Bool, ibuflen::Int, obuflen::Int) = begin
        io = new(py, own, text, false, ibuflen, UInt8[], obuflen, UInt8[])
        finalizer(pyio_finalize!, io)
    end
end
export PyIO

function PyIO(o; own::Bool = false, text::Union{Missing,Bool} = missing, buflen::Integer = 4096, ibuflen::Integer = buflen, obuflen::Integer = buflen)
    if text === missing
        text = pyhasattr(o, "encoding")
    end
    buflen = convert(Int, buflen)
    buflen > 0 || error("buflen must be positive")
    ibuflen = convert(Int, ibuflen)
    ibuflen > 0 || error("ibuflen must be positive")
    obuflen = convert(Int, obuflen)
    obuflen > 0 || error("obuflen must be positive")
    PyIO(Val(:new), Py(o), own, text, ibuflen, obuflen)
end

pyio_finalize!(io::PyIO) = begin
    C.CTX.is_initialized || return
    io.own ? close(io) : flush(io)
    pydel!(io.py)
    return
end

ispy(io::PyIO) = true
getpy(io::PyIO) = io.py
pydel!(io::PyIO) = (finalize(io); nothing)

pyconvert_rule_io(::Type{PyIO}, x::Py) = pyconvert_return(PyIO(x))

"""
    PyIO(f, x; ...)

Equivalent to `f(PyIO(x; ...))` except the stream is automatically flushed or closed according to `own`.

For use in a `do` block, as in
```
PyIO(x, ...) do io
    ...
end
```
"""
function PyIO(f::Function, o; opts...)
    io = PyIO(o; opts...)
    try
        return f(io)
    finally
        pydel!(io)
    end
end

# If obuf is non-empty, write it to the underlying stream.
function putobuf(io::PyIO)
    if !isempty(io.obuf)
        data = io.text ? pystr_fromUTF8(io.obuf) : pybytes(io.obuf)
        pydel!(@py io.write(data))
        pydel!(data)
        empty!(io.obuf)
    end
    return
end

# If ibuf is empty, read some more from the underlying stream.
# After this call, if ibuf is empty then we are at EOF.
# TODO: in binary mode, `io.readinto()` to avoid copying data
function getibuf(io::PyIO)
    if isempty(io.ibuf)
        data = @py io.read(@jl io.ibuflen)
        if io.text
            append!(io.ibuf, pystr_asUTF8vector(data))
        else
            append!(io.obuf, pybytes_asvector(data))
        end
        pydel!(data)
    end
    return
end

function Base.flush(io::PyIO)
    putobuf(io)
    pydel!(@py io.flush())
    return
end

function Base.close(io::PyIO)
    flush(io)
    pydel!(@py io.close())
    return
end

function Base.eof(io::PyIO)
    if io.eof
        true
    elseif !isempty(io.ibuf)
        false
    else
        getibuf(io)
        io.eof = isempty(io.ibuf)
    end
end

Base.fd(io::PyIO) = pyconvert_and_del(Int, @py io.fileno())

Base.isreadable(io::PyIO) = pyconvert_and_del(Bool, @py io.readable())

Base.iswritable(io::PyIO) = pyconvert_and_del(Bool, @py io.writable())

Base.isopen(io::PyIO) = !pyconvert_and_del(Bool, @py io.closed)

function Base.unsafe_write(io::PyIO, ptr::Ptr{UInt8}, n::UInt)
    ntodo = n
    while true
        nroom = max(0, io.obuflen - length(io.obuf))
        if ntodo < nroom
            append!(io.obuf, unsafe_wrap(Array, ptr, ntodo))
            return n
        else
            append!(io.obuf, unsafe_wrap(Array, ptr, nroom))
            putobuf(io)
            ptr += nroom
            ntodo -= nroom
        end
    end
end

function Base.write(io::PyIO, c::UInt8)
    push!(io.obuf, c)
    if length(io.obuf) ≥ io.obuflen
        putobuf(io)
    end
    return
end

function Base.unsafe_read(io::PyIO, ptr::Ptr{UInt8}, n::UInt)
    ntodo = n
    while true
        navail = length(io.ibuf)
        if ntodo ≤ navail
            unsafe_wrap(Array, ptr, ntodo) .= splice!(io.ibuf, 1:ntodo)
            return
        else
            unsafe_wrap(Array, ptr, navail) .= io.ibuf
            ptr += navail
            ntodo -= navail
            empty!(io.ibuf)
            eof(io) && throw(EOFError())
        end
    end
    return
end

function Base.read(io::PyIO, ::Type{UInt8})
    eof(io) && throw(EOFError())
    popfirst!(io.ibuf)
end

function Base.seek(io::PyIO, pos::Integer)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    pydel!(@py io.seek(pos))
    return io
end

function Base.truncate(io::PyIO, pos::Integer)
    seek(io, position(io))
    pydel!(@py io.truncate(pos))
    return io
end

function Base.seekstart(io::PyIO)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    pydel!(@py io.seek(0))
    return io
end

function Base.seekend(io::PyIO)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    pydel!(@py io.seek(0, 2))
    return io
end

function Base.skip(io::PyIO, n::Integer)
    putobuf(io)
    if io.text
        if n ≥ 0
            read(io, n)
        else
            error("`skip(io, n)` for text PyIO streams only implemented for positive `n`")
        end
    else
        if 0 ≤ n ≤ io.ibuflen
            read(io, n)
        else
            pydel!(@py io.seek(@jl(n - length(io.ibuf)), 1))
            empty!(io.ibuf)
            io.eof = false
        end
    end
    return io
end

function Base.position(io::PyIO)
    putobuf(io)
    if io.text
        if isempty(io.ibuf)
            return pyconvert_and_del(Int, @py io.tell())
        else
            error("`position(io)` text PyIO streams only implemented for empty input buffer (e.g. do `read(io, length(io.ibuf))` first)")
        end
    else
        return pyconvert_and_del(Int, @py io.tell()) - length(io.ibuf)
    end
end
