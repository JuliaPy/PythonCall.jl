"""
    PyIO(x; own=false, text=missing, line_buffering=false, buflen=4096)

Wrap the Python IO stream `x` as a Julia IO stream.

When this goes out of scope and is finalized, it is automatically flushed. If `own=true` then it is also closed.

If `text=false` then `x` must be a binary stream and arbitrary binary I/O is possible.
If `text=true` then `x` must be a text stream and only UTF-8 must be written (i.e. use `print` not `write`).
If `text` is not specified then it is chosen automatically.
If `x` is a text stream and you really need a binary stream, then often `PyIO(x.buffer)` will work.

If `line_buffering=true` then output is flushed at each line.

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
    # true to flush whenever '\n' or '\r' is encountered
    line_buffering::Bool
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof::Bool
    # input buffer
    ibuflen::Int
    ibuf::Vector{UInt8}
    # output buffer
    obuflen::Int
    obuf::Vector{UInt8}

    function PyIO(x; own::Bool = false, text::Union{Missing,Bool} = missing, buflen::Integer = 4096, ibuflen::Integer = buflen, obuflen::Integer = buflen, line_buffering::Bool=false)
        if text === missing
            text = pyhasattr(x, "encoding")
        end
        buflen = convert(Int, buflen)
        buflen > 0 || error("buflen must be positive")
        ibuflen = convert(Int, ibuflen)
        ibuflen > 0 || error("ibuflen must be positive")
        obuflen = convert(Int, obuflen)
        obuflen > 0 || error("obuflen must be positive")
        new(Py(x), own, text, line_buffering, false, ibuflen, UInt8[], obuflen, UInt8[])
    end
end
export PyIO

pyio_finalize!(io::PyIO) = begin
    C.CTX[].is_initialized || return
    io.own ? close(io) : flush(io)
    return
end

ispy(io::PyIO) = true
Py(io::PyIO) = io.py

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

Base.fd(io::PyIO) = pyconvert(Int, @py io.fileno())

Base.isreadable(io::PyIO) = pyconvert(Bool, @py io.readable())

Base.iswritable(io::PyIO) = pyconvert(Bool, @py io.writable())

Base.isopen(io::PyIO) = !pyconvert(Bool, @py io.closed)

function Base.unsafe_write(io::PyIO, ptr::Ptr{UInt8}, n::UInt)
    ntodo = n
    while ntodo > 0
        nroom = max(0, io.obuflen - length(io.obuf))
        if ntodo < nroom
            buf = unsafe_wrap(Array, ptr, ntodo)
            if io.line_buffering
                i = findlast(∈((0x0A, 0x0D)), buf)
                if i === nothing
                    append!(io.obuf, buf)
                else
                    append!(io.obuf, unsafe_wrap(Array, ptr, i))
                    putobuf(io)
                    append!(io.obuf, unsafe_wrap(Array, ptr+i, ntodo-i))
                end
            else
                append!(io.obuf, buf)
            end
            break
        else
            buf = unsafe_wrap(Array, ptr, nroom)
            append!(io.obuf, buf)
            putobuf(io)
            ptr += nroom
            ntodo -= nroom
        end
    end
    return n
end

function Base.write(io::PyIO, c::UInt8)
    push!(io.obuf, c)
    if (length(io.obuf) ≥ io.obuflen) || (io.line_buffering && (c == 0x0A || c == 0x0D))
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
            return pyconvert(Int, @py io.tell())
        else
            error("`position(io)` text PyIO streams only implemented for empty input buffer (e.g. do `read(io, length(io.ibuf))` first)")
        end
    else
        return pyconvert(Int, @py io.tell()) - length(io.ibuf)
    end
end
