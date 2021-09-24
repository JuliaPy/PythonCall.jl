struct PyPlotFigure
    py::Py
end
export PyPlotFigure

ispy(::PyPlotFigure) = true
getpy(fig::PyPlotFigure) = fig.py

Base.show(io::IO, mime::MIME"image/png", fig::PyPlotFigure) = _pyplot_show(io, mime, fig, "png")
Base.show(io::IO, mime::MIME"image/jpeg", fig::PyPlotFigure) = _pyplot_show(io, mime, fig, "jpeg")
Base.show(io::IO, mime::MIME"image/tiff", fig::PyPlotFigure) = _pyplot_show(io, mime, fig, "tiff")
Base.show(io::IO, mime::MIME"image/svg+xml", fig::PyPlotFigure) = _pyplot_show(io, mime, fig, "svg")
Base.show(io::IO, mime::MIME"application/pdf", fig::PyPlotFigure) = _pyplot_show(io, mime, fig, "pdf")

Base.showable(::MIME"image/png", fig::PyPlotFigure) = _pyplot_showable(fig, "png")
Base.showable(::MIME"image/jpeg", fig::PyPlotFigure) = _pyplot_showable(fig, "jpeg")
Base.showable(::MIME"image/tiff", fig::PyPlotFigure) = _pyplot_showable(fig, "tiff")
Base.showable(::MIME"image/svg+xml", fig::PyPlotFigure) = _pyplot_showable(fig, "svg")
Base.showable(::MIME"application/pdf", fig::PyPlotFigure) = _pyplot_showable(fig, "pdf")

function _pyplot_bytes(fig::Py, format::String)
    buf = pyimport("io").BytesIO()
    fig.savefig(buf, format=format)
    return pyconvert(Vector{UInt8}, buf.getvalue())
end

function _pyplot_show(io::IO, mime::MIME, fig::PyPlotFigure, format::String)
    try
        write(io, _pyplot_bytes(fig.py, format))
        return
    catch exc
        if exc isa PyException
            throw(MethodError(show, (io, mime, fig)))
        else
            rethrow()
        end
    end
end

function _pyplot_showable(fig::PyPlotFigure, format::String)
    try
        _pyplot_bytes(fig.py, format)
        return true
    catch exc
        if exc isa PyException
            return false
        else
            rethrow()
        end
    end
end

"""
    pyplot_gcf(; close=true)

Get the current matplotlib/pyplot/seaborn/etc figure as an object displayable by Julia's display mechanism.

If `close` is true, the figure is also closed.
"""
function pyplot_gcf(; close::Bool=true)
    plt = pyimport("matplotlib.pyplot")
    fig = plt.gcf()
    close && plt.close(fig)
    PyPlotFigure(fig)
end
export pyplot_gcf

"""
    pyplot_show([fig]; close=true, [format])

Show the matplotlib/pyplot/seaborn/etc figure `fig` using Julia's display mechanism.

If `fig` is not given, then all open figures are shown.

If `close` is true, the figure is also closed.

The `format` specifies the file format of the generated image.
If not given, it is decided by the display.
"""
function pyplot_show(fig; close::Bool = true, format::Union{String,Nothing} = nothing)
    plt = pyimport("matplotlib.pyplot")
    fig = Py(fig)
    if !pyisinstance(fig, plt.Figure)
        fig = plt.figure(fig)
    end
    fig = PyPlotFigure(fig)
    if format === nothing
        display(fig)
    elseif format == "png"
        display(MIME("image/png"), fig)
    elseif format in ("jpg", "jpeg")
        display(MIME("image/jpeg"), fig)
    elseif format in ("tif", "tiff")
        display(MIME("image/tiff"), fig)
    elseif format == "svg"
        display(MIME("image/svg+xml"), fig)
    elseif format == "pdf"
        display(MIME("application/pdf"), fig)
    else
        error("invalid format: $format")
    end
    close && plt.close(fig)
    return
end

function pyplot_show(; opts...)
    plt = pysysmodule.modules.get("matplotlib.pyplot", nothing)
    if !pyisnone(plt)
        for fig in plt.get_fignums()
            pyplot_show(fig; opts...)
        end
    end
end
export pyplot_show

function init_matplotlib()
    @require IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a" begin
        IJulia.push_postexecute_hook() do
            CONFIG.auto_pyplot_show && pyplot_show()
            nothing
        end
    end
end
