struct IPythonDisplay <: AbstractDisplay end

Base.display(d::IPythonDisplay, m::MIME, @nospecialize(x)) =
    try
        buf = IOBuffer()
        show(buf, m, x)
        data = take!(buf)
        dict = pydict()
        dict[string(m)] = istextmime(m) ? pystr(data) : pybytes(data)
        @py `sys.modules["IPython"].display.display($dict, raw=True)`
    catch
        throw(MethodError(display, (d, m, x)))
    end

Base.display(d::IPythonDisplay, @nospecialize(x)) = begin
    if ispyref(x)
        @py `sys.modules["IPython"].display.display($x)`
        return
    end
    buf = IOBuffer()
    dict = pydict()
    for m in C.mimes_for(x)
        try
            show(buf, MIME(m), x)
        catch
            continue
        end
        data = take!(buf)
        dict[m] = istextmime(m) ? pystr(data) : pybytes(data)
    end
    length(dict) == 0 && throw(MethodError(display, (d, x)))
    try
        @py `sys.modules["IPython"].display.display($dict, raw=True)`
    catch
        throw(MethodError(display, (d, x)))
    end
end
