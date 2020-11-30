pymatplotlib = PyLazyObject(() -> pyimport("matplotlib"))
pyplot = PyLazyObject(() -> pyimport("matplotlib.pyplot"))

"""
    pyplotshow([mime], [fig]; close=true, format=CONFIG.pyplotshowformat)

Show the matplotlib/pyplot/seaborn/etc figure `fig`, or all open figures if not given.

If `close` is true, the figure is also closed.

The MIME type can be specified by `mime`. The file format used can be specified by `format`, the default being configurable.

If `CONFIG.pyplotautoshow` is true, then this is automatically called each time a notebook cell is evaluated.
"""
function pyplotshow(mime::MIME"text/html", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="png")
    display(mime, HTML("""<img src="data:image/png;base64,$(base64encode(take!(io)))" />"""))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(mime::MIME"image/png", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    format in ("any", "png") || error("invalid format")
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="png")
    display(mime, take!(io))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(mime::MIME"image/jpeg", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    format in ("any", "jpeg", "jpg") || error("invalid format")
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="jpeg")
    display(mime, take!(io))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(mime::MIME"image/tiff", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    format in ("any", "tif", "tiff") || error("invalid format")
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="tiff")
    display(mime, take!(io))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(mime::MIME"image/svg+xml", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    format in ("any", "svg") || error("invalid format")
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="svg")
    display(mime, String(take!(io)))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(mime::MIME"application/pdf", fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    format in ("any", "pdf") || error("invalid format")
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format="pdf")
    display(mime, take!(io))
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(fig; close::Bool=true, format::String=CONFIG.pyplotshowformat)
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    okfmts = fig.canvas.get_supported_filetypes()
    format == "any" || format in okfmts || error("format $(repr(format)) not supported by this backend")
    for (mime,fmts) in ((MIME("image/png"), ("png",)), (MIME("image/jpeg"), ("jpeg","jpg")), (MIME("image/tiff"), ("tiff","tif")), (MIME("image/svg+xml"), ("svg",)), (MIME("text/html"), ("png",)), (MIME("application/pdf"), ("pdf",)))
        if displayable(mime)
            for fmt in fmts
                if format in ("any", fmt) && fmt in okfmts
                    return pyplotshow(mime, fig, close=close, format=format)
                end
            end
        end
    end
    # no compatible format
    if format == "any"
        display(fig)
        close && pyplot.close(fig)
        return
    end
    error("can't display format $(repr(format)) on this display")
end
function pyplotshow(mime::MIME; opts...)
    for fig in pyplot.get_fignums()
        pyplotshow(mime, fig; opts...)
    end
end
function pyplotshow(; opts...)
    for fig in pyplot.get_fignums()
        pyplotshow(fig; opts...)
    end
end
export pyplotshow
