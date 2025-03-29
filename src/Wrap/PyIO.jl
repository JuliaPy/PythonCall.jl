pyio_finalize!(io::PyIO) = begin
    C.CTX.is_initialized || return
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
        pydel!(io.py)
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
            append!(io.ibuf, pybytes_asvector(data))
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
                    append!(io.obuf, unsafe_wrap(Array, ptr + i, ntodo - i))
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
            error(
                "`position(io)` text PyIO streams only implemented for empty input buffer (e.g. do `read(io, length(io.ibuf))` first)",
            )
        end
    else
        return pyconvert(Int, @py io.tell()) - length(io.ibuf)
    end
end
