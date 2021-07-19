"""
    pyplot_show([fig]; close=true, [format])

Show the matplotlib/pyplot/seaborn/etc figure `fig`, or all open figures if not given, using Julia's display mechanism.

If `close` is true, the figure is also closed.

The `format` specifies the file format of the generated image.
By default this is `pyplot.rcParams["savefig.format"]` or `"png"`.
It can be one of `"png"`, `"jpg"`, `"jpeg"`, `"tif"`, `"tiff"`, `"svg"` or `"pdf"`.
"""
function pyplot_show(fig; close::Bool = true, format::String = "")
    plt = pyimport("matplotlib.pyplot")
    io = pyimport("io")
    if !pyisinstance(fig, plt.Figure)
        fig = plt.figure(fig)
    end
    buf = io.BytesIO()
    if isempty(format)
        format = pyconvert(String, plt.rcParams.get("savefig.format", "png"))
    end
    if format ∉ ("png", "jpg", "jpeg", "tif", "tiff", "svg", "pdf")
        error("invalid format: $format")
    end
    fig.savefig(buf, format=format)
    data = pyconvert(Vector{UInt8}, buf.getvalue())
    if close
        plt.close(fig)
    end
    if format == "png"
        display(MIME("image/png"), data)
    elseif format in ("jpg", "jpeg")
        display(MIME("image/jpeg"), data)
    elseif format in ("tif", "tiff")
        display(MIME("image/tiff"), data)
    elseif format == "svg"
        display(MIME("image/svg+xml"), String(data))
    elseif format == "pdf"
        display(MIME("application/pdf"), data)
    else
        @assert false
    end
    nothing
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
