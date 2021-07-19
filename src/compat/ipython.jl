struct IPythonDisplay <: AbstractDisplay end

function Base.display(d::IPythonDisplay, m::MIME, @nospecialize(x))
    ipy = pyimport("IPython")
    buf = IOBuffer()
    dict = pydict()
    try
        show(buf, m, x)
    catch
        throw(MethodError(display, (d, m, x)))
    end
    data = take!(buf)
    dict[string(m)] = istextmime(m) ? pystr_fromUTF8(data) : pybytes(data)
    ipy.display.display(dict, raw=true)
    return
end

function Base.display(d::IPythonDisplay, @nospecialize(x))
    ipy = pyimport("IPython")
    if ispy(x)
        ipy.display.display(x)
        return
    end
    buf = IOBuffer()
    dict = pydict()
    for m in Utils.mimes_for(x)
        try
            show(buf, MIME(m), x)
        catch
            continue
        end
        data = take!(buf)
        dict[m] = istextmime(m) ? pystr_fromUTF8(data) : pybytes(data)
    end
    length(dict) == 0 && throw(MethodError(display, (d, x)))
    ipy.display.display(dict, raw=true)
    return
end

function init_ipython()
    # EXPERIMENTAL: IPython integration
    if C.CTX.is_embedded && CONFIG.auto_ipython_display
        is_ipython = ("IPython" in pysysmodule.modules) && !pyisnone(pysysmodule.modules["IPython"].get_ipython())
        if is_ipython
            # Set `Base.stdout` to `sys.stdout` and ensure it is flushed after each execution
            @eval Base stdout = $(PyIO(pysysmodule.stdout))
            pysysmodule.modules["IPython"].get_ipython().events.register("post_execute", pycallback(() -> (flush(Base.stdout); nothing)))
            # set displays so that Base.display() renders in ipython
            pushdisplay(TextDisplay(Base.stdout))
            pushdisplay(IPythonDisplay())
        end
    end
end
