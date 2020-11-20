"""
    PyIO(o; text=missing)

Wrap the Python byte-based IO stream `o` as a Julia IO stream.
"""
struct PyIO <: IO
    o :: PyObject
end
export PyIO

pyobject(io::PyIO) = io.o

Base.flush(io::PyIO) = (io.o.flush(); nothing)

Base.close(io::PyIO) = (io.o.close(); nothing)

Base.position(io::PyIO) = io.o.tell().jl!i

Base.seek(io::PyIO, n::Integer) = (io.o.seek(pyint(n)); io)

Base.seekstart(io::PyIO) = (io.o.seek(0); io)

Base.seekend(io::PyIO) = (io.o.seek(0,2); io)

Base.skip(io::PyIO, n::Integer) = (io.o.seek(pyint(n), 1); io)

Base.fd(io::PyIO) = io.o.fileno().jl!i

Base.isreadable(io::PyIO) = io.o.readable().jl!b

Base.iswritable(io::PyIO) = io.o.writable().jl!b

Base.isopen(io::PyIO) = !(io.o.closed.jl!b)

Base.truncate(io::PyIO, n::Integer) = (io.o.truncate(n); nothing)

function Base.readbytes!(io::PyIO, b::AbstractVector{UInt8}, n=length(b))
    n = min(n, length(b))
    pb = pyjl(view(b, firstindex(b):firstindex(b)+n-1))
    if C.PyObject_CheckBuffer(pb) == 0
        b2 = Vector{UInt8}(undef, n)
        nr = io.o.readinto(b2).jl!i
        copyto!(b, b2)
        nr
    else
        io.o.readinto(pb).jl!i
    end
end

Base.read(io::PyIO, ::Type{UInt8}) = read(io, 1)[1]

Base.unsafe_write(io::PyIO, ptr::Ptr{UInt8}, n::UInt) =
    io.o.write(unsafe_wrap(Array{UInt8}, ptr, n)).jl!i

Base.write(io::PyIO, c::UInt8) =
    write(io, [c])

Base.write(io::PyIO, x::Array{UInt8}) =
    GC.@preserve x unsafe_write(io, pointer(x), length(x))

Base.eof(io::PyIO) = !pytruth(io.o.peek(1))
