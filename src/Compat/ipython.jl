"""Integration with IPython"""
module IPython

using ...PythonCall
using ...Core: pystr_fromUTF8

"""
    PythonDisplay()

Like TextDisplay() but prints to Python's stdout.
"""
struct PythonDisplay <: AbstractDisplay end

function Base.display(d::PythonDisplay, m::MIME, @nospecialize(x))
    istextmime(m) || throw(MethodError(display, (d, m, x)))
    buf = IOBuffer()
    io = IOContext(buf, :limit => true)
    try
        show(io, m, x)
    catch
        throw(MethodError(display, (d, m, x)))
    end
    data = String(take!(buf))
    pyprint(data)
    return
end

function Base.display(d::PythonDisplay, @nospecialize(x))
    display(d, MIME("text/plain"), x)
end

"""
    IPythonDisplay()

For displaying multimedia with IPython's display mechanism.
"""
struct IPythonDisplay <: AbstractDisplay end

function Base.display(d::IPythonDisplay, m::MIME, @nospecialize(x))
    ipy = pyimport("IPython")
    buf = IOBuffer()
    io = IOContext(buf, :limit => true)
    dict = pydict()
    try
        show(io, m, x)
    catch
        throw(MethodError(display, (d, m, x)))
    end
    data = take!(buf)
    dict[string(m)] = istextmime(m) ? pystr_fromUTF8(data) : pybytes(data)
    ipy.display.display(dict, raw = true)
    return
end

function Base.display(d::IPythonDisplay, @nospecialize(x))
    ipy = pyimport("IPython")
    if ispy(x)
        ipy.display.display(x)
        return
    end
    buf = IOBuffer()
    io = IOContext(buf, :limit => true)
    dict = pydict()
    for m in Utils.mimes_for(x)
        try
            show(io, MIME(m), x)
        catch
            continue
        end
        data = take!(buf)
        dict[m] = istextmime(m) ? pystr_fromUTF8(data) : pybytes(data)
    end
    length(dict) == 0 && throw(MethodError(display, (d, x)))
    ipy.display.display(dict, raw = true)
    return
end

end
