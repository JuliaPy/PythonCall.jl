pymatplotlib = pylazyobject(() -> pyimport("matplotlib"))
pyplot = pylazyobject(() -> pyimport("matplotlib.pyplot"))

"""
    pyplotshow([fig]; close=true, [format])

Show the matplotlib/pyplot/seaborn/etc figure `fig`, or all open figures if not given.

If `close` is true, the figure is also closed.

The `format` specifies the file format of the generated image. By default this is `pyplot.rcParams["savefig.format"]`.
"""
function pyplotshow(fig; close::Bool=true, format::String=pyplot.rcParams.get("savefig.format", "png").jl!s)
    fig = pyisinstance(fig, pyplot.Figure) ? PyObject(fig) : pyplot.figure(fig)
    io = IOBuffer()
    fig.savefig(io, format=format)
    data = take!(io)
    if format == "png"
        display(MIME"image/png"(), data)
    elseif format in ("jpg", "jpeg")
        display(MIME"image/jpeg"(), data)
    elseif format in ("tif", "tiff")
        display(MIME"image/tiff"(), data)
    elseif format == "svg"
        display(MIME"image/svg+xml"(), String(data))
    elseif format == "pdf"
        display(MIME"application/pdf"(), data)
    else
        error("Unsupported format: $(repr(format)) (try one of: png, jpg, jpeg, tif, tiff, svg, xml)")
    end
    close && pyplot.close(fig)
    nothing
end
function pyplotshow(; opts...)
    for fig in pyplot.get_fignums()
        pyplotshow(fig; opts...)
    end
end
export pyplotshow
