"""
    PyIO(o; own=false, text=missing, buflen=4096)

Wrap the Python byte-based IO stream `o` as a Julia IO stream.

When this goes out of scope and is finalized, it is automatically flushed. If `own=true` then it is also closed.

If `text=false` then `o` must be a binary stream and arbitrary binary I/O is possible. If `text=true` then `o` must be a text stream and only UTF-8 must be written (i.e. use `print` not `write`). If `text` is not specified then it is chosen automatically. If `o` is a text stream and you really need a binary stream, then often `PyIO(o.buffer)` will work.

For efficiency, reads and writes are buffered before being sent to `o`. The size of the buffer is `buflen`.
"""
mutable struct PyIO <: IO
    o :: PyObject
    # true to close the file automatically
    own :: Bool
    # true if `o` is text, false if binary
    text :: Bool
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof :: Bool
    # input buffer
    ibuflen :: Int
    ibuf :: Vector{UInt8}
    # output buffer
    obuflen :: Int
    obuf :: Vector{UInt8}

    function PyIO(o::AbstractPyObject; own::Bool=false, text::Union{Missing,Bool}=missing, buflen::Integer=4096, ibuflen=buflen, obuflen=buflen)
        io = new(PyObject(o), own, text===missing ? pyisinstance(o, pyiomodule.TextIOBase) : text, false, ibuflen, UInt8[], obuflen, UInt8[])
        finalizer(io) do io
            io.own ? close(io) : flush(io)
        end
        io
    end
end
export PyIO

"""
    PyIO(f, o; ...)

Equivalent to `f(PyIO(o; ...))` except the stream is automatically flushed or closed according to `own`.

For use in a `do` block, like `PyIO(o, ...) do io; ...; end`.
"""
function PyIO(f::Function, o; opts...)
    io = PyIO(o; opts...)
    try
        return f(io)
    finally
        io.own ? close(io) : flush(io)
    end
end

pyobject(io::PyIO) = io.o

# If obuf is non-empty, write it to the underlying stream.
function putobuf(io::PyIO)
    if !isempty(io.obuf)
        io.text ? io.o.write(String(io.obuf)) : io.o.write(io.obuf)
        empty!(io.obuf)
    end
    nothing
end

# If ibuf is empty, read some more from the underlying stream.
# After this call, if ibuf is empty then we are at EOF.
function getibuf(io::PyIO)
    if isempty(io.ibuf)
        if io.text
            append!(io.ibuf, PyVector{UInt8,UInt8,false,true}(pystr_asutf8string(io.o.read(io.ibuflen))))
        else
            resize!(io.ibuf, io.ibuflen)
            n = io.o.readinto(io.ibuf).jl!i
            resize!(io.ibuf, n)
        end
    end
    nothing
end

function Base.flush(io::PyIO)
    putobuf(io)
    io.o.flush()
    nothing
end

Base.close(io::PyIO) = (flush(io); io.o.close(); nothing)

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

Base.fd(io::PyIO) = io.o.fileno().jl!i

Base.isreadable(io::PyIO) = io.o.readable().jl!b

Base.iswritable(io::PyIO) = io.o.writable().jl!b

Base.isopen(io::PyIO) = !io.o.closed.jl!b

function Base.unsafe_write(io::PyIO, ptr::Ptr{UInt8}, n::UInt)
    while true
        m = max(0, io.obuflen - length(io.obuf))
        if n < m
            append!(io.obuf, unsafe_wrap(Array, ptr, n))
            return
        else
            append!(io.obuf, unsafe_wrap(Array, ptr, m))
            putobuf(io)
            ptr += m
            n -= m
        end
    end
end

function Base.write(io::PyIO, c::UInt8)
    push!(io.obuf, c)
    if length(io.obuf) ≥ io.obuflen
        putobuf(io)
    end
end

function Base.unsafe_read(io::PyIO, ptr::Ptr{UInt8}, n::UInt)
    while true
        m = length(io.ibuf)
        if n ≤ m
            unsafe_wrap(Array, ptr, n) .= splice!(io.ibuf, 1:n)
            return
        else
            unsafe_wrap(Array, ptr, m) .= io.ibuf
            ptr += m
            n -= m
            empty!(io.ibuf)
            eof(io) && throw(EOFError())
        end
    end
end

function Base.read(io::PyIO, ::Type{UInt8})
    eof(io) && throw(EOFError())
    popfirst!(io.ibuf)
end

function seek(io::PyIO, pos::Integer)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    io.o.seek(pos, 0)
    io
end

function truncate(io::PyIO, pos::Integer)
    seek(io, position(io))
    io.o.truncate(pos)
    io
end

function seekstart(io::PyIO)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    io.o.seek(0, 0)
    io
end

function seekend(io::PyIO)
    putobuf(io)
    empty!(io.ibuf)
    io.eof = false
    io.o.seek(0, 2)
    io
end

function skip(io::PyIO, n::Integer)
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
            io.o.seek(n - length(io.ibuf), 1)
            empty!(io.ibuf)
            io.eof = false
        end
    end
end

function position(io::PyIO)
    putobuf(io)
    if io.text
        if isempty(io.ibuf)
            io.o.position().jl!i
        else
            error("`position(io)` text PyIO streams only implemented for empty input buffer (e.g. do `read(io, length(io.ibuf))` first)")
        end
    else
        io.o.position().jl!i - length(io.ibuf)
    end
end
