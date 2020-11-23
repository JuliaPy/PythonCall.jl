"""
    PyByteIO(o; buflen=4096)

Wrap the Python byte-based IO stream `o` as a Julia IO stream.

For efficiency, reads and writes are buffered before being sent to `o`. The size of the buffer is `buflen`.
"""
mutable struct PyByteIO <: IO
    o :: PyObject
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof :: Bool
    # input buffer
    ibuf :: Vector{UInt8}
    ibufpos :: Int
    ibufused :: Int
    # output buffer
    obuf :: Vector{UInt8}
    obufused :: Int
end
PyByteIO(o::AbstractPyObject; buflen=4096, ibuflen=buflen, obuflen=buflen) =
    PyByteIO(PyObject(o), false, Vector{UInt8}(undef, ibuflen), 0, 0, Vector{UInt8}(undef, obuflen), 0)
export PyByteIO

pyobject(io::PyByteIO) = io.o

# write the contents of the out buffer
function writeobuf(io::PyByteIO)
    if io.obufused > 0
        n = io.o.write(view(io.obuf, 1:io.obufused)).jl!i
        @assert io.obufused == n
        io.obufused = 0
    end
    nothing
end

# if possible, read more into the in buffer
# if the in buffer is empty after this call then we are at EOF
function readibuf(io::PyByteIO)
    if io.ibufpos == io.ibufused
        # we have consumed the whole buffer, start over
        io.ibufpos = io.ibufused = 0
    end
    if io.ibufused < length(io.ibuf)
        # still space for a read at the end of the buffer
        n = io.o.readinto(@view io.ibuf[io.ibufused+1:end]).jl!i
        io.ibufused += n
    end
    nothing
end

function Base.flush(io::PyByteIO)
    writeobuf(io)
    io.o.flush()
    nothing
end

Base.close(io::PyByteIO) = (flush(io); io.o.close(); nothing)

function Base.eof(io::PyByteIO)
    if io.eof
        true
    elseif io.ibufpos < io.ibufused
        false
    else
        readibuf(io)
        io.eof = io.ibufpos == io.ibufused
    end
end

Base.fd(io::PyByteIO) = io.o.fileno().jl!i

Base.isreadable(io::PyByteIO) = io.o.readable().jl!b

Base.iswritable(io::PyByteIO) = io.o.writable().jl!b

Base.isopen(io::PyByteIO) = !io.o.closed.jl!b

function Base.unsafe_write(io::PyByteIO, ptr::Ptr{UInt8}, n::UInt)
    m = length(io.obuf) - io.obufused
    if n ≤ m
        # it fits in the buffer
        io.obuf[io.obufused+1:io.obufused+n] .= unsafe_wrap(Array, ptr, n)
        io.obufused += n
        if io.obufused == length(io.obuf)
            writeobuf(io)
        end
    else
        # first, fill up the buffer and write it
        io.obuf[io.obufused+1:io.obufused+m] .= unsafe_wrap(Array, ptr, m)
        io.obufused += m
        writeobuf(io)
        # still have n-m bytes to write
        ptr += m
        n -= m
        if n ≤ length(io.obuf)
            # the rest fits in the buffer
            unsafe_write(io, ptr, n)
        else
            # otherwise, write directly
            n′ = io.o.write(unsafe_wrap(Array, ptr, n)).jl!i
            @assert n == n′
        end
    end
    nothing
end

function Base.write(io::PyByteIO, c::UInt8)
    if io.obufused == length(io.obuf)
        writeobuf(io)
    end
    @assert io.obufused < length(io.obuf) # TODO: allow unbuffered output
    io.obuf[io.obufused+1] = c
    io.obufused += 1
    if io.obufused == length(io.obuf)
        writeobuf(io)
    end
end

function Base.unsafe_read(io::PyByteIO, ptr::Ptr{UInt8}, n::UInt)
    m = io.ibufused - io.ibufpos
    if n ≤ m
        # already have n bytes in the buffer
        unsafe_wrap(Array, ptr, n) .= @view io.ibuf[io.ibufpos+1:io.ibufpos+n]
        io.pos += n
        io.ibufpos += n
    else
        # first copy everything out of the buffer
        unsafe_wrap(Array, ptr, n) .= @view io.ibuf[io.ibufpos+1:io.ibufpos+m]
        io.pos += m
        io.ibufpos += m
        # still have n-m bytes to read
        ptr += m
        n -= m
        if n ≤ length(io.ibuf)
            # if the read fits in one more buffer, read a whole buffer worth and try again
            readibuf(io)
            unsafe_read(io, ptr, n)
        else
            # otherwise, read directly
            n′ = io.o.readinto(unsafe_wrap(Array, ptr, n)).jl!i
            @assert n′ == n
            io.pos += n
        end
    end
    nothing
end

function Base.read(io::PyByteIO, ::Type{UInt8})
    eof(io) && throw(EOFError())
    r = io.ibuf[io.ibufpos+1]
    io.pos += 1
    io.ibufpos += 1
    r
end


"""
    PyTextIO(o; buflen=4096)

Wrap the Python text-based IO stream `o` as a Julia IO stream.

Since encoding of `o` is respected, only the textual operations are supported: `print(io, ...)`, `read(io, String)`, `read(io, Char)`, `readuntil(io, delim)`, `readline(io)`.

For efficiency, reads and writes are buffered before being sent to `o`. The size of the buffer is `buflen`.
"""
mutable struct PyTextIO <: IO
    o :: PyObject
    # true if we are definitely at the end of the file; false if we are not or don't know
    eof :: Bool
    # input buffer
    ibuflen :: Int
    ibuf :: String
    ibufpos :: Int
    # output buffer
    obuflen :: Int
    obuf :: Vector{String}
    obufused :: Int
end
PyTextIO(o::AbstractPyObject; buflen=4096, ibuflen=buflen, obuflen=buflen) =
    PyTextIO(PyObject(o), false, ibuflen, "", 1, obuflen, String[], 0)
export PyTextIO

pyobject(io::PyTextIO) = io.o

# write the contents of the out buffer
function writeobuf(io::PyTextIO)
    if io.obufused > 0
        io.o.write(join(io.obuf))
        empty!(io.obuf)
        io.obufused = 0
    end
    nothing
end

# if possible, read more into the in buffer
# if the in buffer is empty after this call then we are at EOF
function readibuf(io::PyTextIO)
    if io.ibufpos > lastindex(io.ibuf)
        io.ibuf = pystr_asjuliastring(io.o.read(io.ibuflen))
        io.ibufpos = 1
    end
    nothing
end

function Base.flush(io::PyTextIO)
    writeobuf(io)
    io.o.flush()
    nothing
end

Base.close(io::PyTextIO) = (flush(io); io.o.close(); nothing)

function Base.eof(io::PyTextIO)
    if io.eof
        true
    elseif io.ibufpos ≤ lastindex(io.ibuf)
        false
    else
        readibuf(io)
        io.eof = io.ibufpos > lastindex(io.ibuf)
    end
end

Base.fd(io::PyTextIO) = io.o.fileno().jl!i

Base.isreadable(io::PyTextIO) = io.o.readable().jl!b

Base.iswritable(io::PyTextIO) = io.o.writable().jl!b

Base.isopen(io::PyTextIO) = !io.o.closed.jl!b

function Base.print(io::PyTextIO, _x)
    x = convert(String, string(_x))
    push!(io.obuf, x)
    io.obufused += sizeof(x)
    if io.obufused ≥ io.obuflen
        writeobuf(io)
    end
    nothing
end
Base.print(io::PyTextIO, x::Union{String, SubString{String}}) = invoke(print, Tuple{PyTextIO, Any}, io, x)
Base.print(io::PyTextIO, x::Char) = invoke(print, Tuple{PyTextIO, Any}, io, x)

function Base.read(io::PyTextIO, ::Type{String})
    r = io.ibuf[io.ibufpos:end] * pystr_asjuliastring(io.o.read())
    io.ibuf = ""
    io.ibufpos = 1
    r
end

function Base.read(io::PyTextIO, ::Type{Char})
    eof(io) && throw(EOFError())
    @assert io.ibufpos ≤ lastindex(io.ibuf)
    r = io.ibuf[io.ibufpos]
    io.ibufpos = nextind(io.ibuf, io.ibufpos)
    r
end

function Base.readuntil(io::PyTextIO, delim::Char; keep::Bool=false)
    xs = String[]
    while true
        eof(io) && throw(EOFError())
        pos = findnext(delim, io.ibuf, io.ibufpos)
        if pos === nothing
            push!(xs, io.ibuf[io.ibufpos:end])
            io.ibufpos = 1
            io.ibuf = ""
            readibuf(io)
        else
            push!(xs, io.ibuf[io.ibufpos:(keep ? pos : prevind(io.ibuf, pos))])
            io.ibufpos = nextind(io.ibuf, pos)
            break
        end
    end
    join(xs)
end

function Base.readuntil(io::PyTextIO, delim::String; keep::Bool=false)
    if length(delim) == 0
        ""
    elseif length(delim) == 1
        readuntil(io, delim[1], keep=keep)
    else
        error("not implemented")
    end
end

function Base.readline(io::PyTextIO; keep::Bool=false)
    line = readuntil(io, '\n', keep=keep)
    !keep && endswith(line, '\r') ? line[1:end-1] : line
end
