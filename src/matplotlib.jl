"""
    pyplotshow([fig]; close=true, [format])

Show the matplotlib/pyplot/seaborn/etc figure `fig`, or all open figures if not given.

If `close` is true, the figure is also closed.

The `format` specifies the file format of the generated image. By default this is `pyplot.rcParams["savefig.format"]`.
"""
function pyplotshow(fig; close::Bool = true, format::String = "")
    @py ```
    import matplotlib.pyplot as plt, io
    fig = $fig
    if not isinstance(fig, plt.Figure):
        fig = plt.figure(fig)
    buf = io.BytesIO()
    format = $format
    if not format:
        format = plt.rcParams.get("savefig.format", "png")
    if format not in ["png", "jpg", "jpeg", "tif", "tiff", "svg", "pdf"]:
        raise ValueError("Unsupported format: {}".format(format))
    fig.savefig(buf, format=format)
    $(data::Vector{UInt8}) = buf.getvalue()
    if $close:
        plt.close(fig)
    $(format::String) = format
    ```
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
        error()
    end
    nothing
end
function pyplotshow(; opts...)
    @py ```
    import sys
    plt = sys.modules.get("matplotlib.pyplot", None)
    $(fignums::Vector{Int}) = [] if plt is None else plt.get_fignums()
    ```
    for fig in fignums
        pyplotshow(fig; opts...)
    end
end
export pyplotshow
