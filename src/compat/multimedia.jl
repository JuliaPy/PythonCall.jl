### Extensible system for multimedia display of Python objects

const PYSHOW_RULES = Function[]

function pyshow_add_rule(rule::Function)
    push!(PYSHOW_RULES, rule)
    return
end

function pyshow(io::IO, mime::MIME, x)
    x_ = Py(x)
    for rule in PYSHOW_RULES
        rule(io, string(mime), x_) && return
    end
    throw(MethodError(show, (io, mime, x_)))
end

function pyshowable(mime::MIME, x)
    x_ = Py(x)
    for rule in PYSHOW_RULES
        rule(devnull, string(mime), x_) && return true
    end
    return false
end

### Particular rules

# x._repr_mimebundle_()
function pyshow_rule_mimebundle(io::IO, mime::String, x::Py)
    try
        ans = x._repr_mimebundle_(include=pylist((mime,)))
        if pyisinstance(ans, pybuiltins.tuple)
            data = ans[0][mime]
        else
            data = ans[mime]
        end
        write(io, pyconvert(Union{String,Vector{UInt8}}, data))
        return true
    catch exc
        if exc isa PyException
            return false
        else
            rethrow()
        end
    end
end

const MIME_TO_REPR_METHOD = Dict(
    "text/plain" => "__repr__",
    "text/html" => "_repr_html_",
    "text/markdown" => "_repr_markdown_",
    "text/json" => "_repr_json_",
    "text/latex" => "_repr_latex_",
    "application/javascript" => "_repr_javascript_",
    "application/pdf" => "_repr_pdf_",
    "image/jpeg" => "_repr_jpeg_",
    "image/png" => "_repr_png_",
    "image/svg+xml" => "_repr_svg_",
)

# x._repr_FORMAT_()
function pyshow_rule_repr(io::IO, mime::String, x::Py)
    method = get(MIME_TO_REPR_METHOD, mime, "")
    isempty(method) && return false
    try
        ans = pygetattr(x, method)()
        if pyisinstance(ans, pybuiltins.tuple)
            data = ans[0]
        else
            data = ans
        end
        write(io, pyconvert(Union{String,Vector{UInt8}}, data))
        return true
    catch exc
        if exc isa PyException
            return false
        else
            rethrow()
        end
    end
end

const MIME_TO_MATPLOTLIB_FORMAT = Dict(
    "image/png" => "png",
    "image/jpeg" => "jpeg",
    "image/tiff" => "tiff",
    "image/svg+xml" => "svg",
    "application/pdf" => "pdf",
)

# x.savefig()
# Requires x to be a matplotlib.pyplot.Figure, or x.figure to be one.
# Closes the underlying figure.
function pyshow_rule_savefig(io::IO, mime::String, x::Py)
    format = get(MIME_TO_MATPLOTLIB_FORMAT, mime, "")
    isempty(format) && return false
    pyhasattr(x, "savefig") || return false
    try
        plt = pysysmodule.modules["matplotlib.pyplot"]
        Figure = plt.Figure
        fig = x
        while !pyisinstance(fig, Figure)
            fig = fig.figure
        end
        buf = pyimport("io").BytesIO()
        x.savefig(buf, format=format)
        data = pyconvert(Vector{UInt8}, buf.getvalue())
        write(io, data)
        plt.close(fig)
        return true
    catch exc
        if exc isa PyException
            return false
        else
            rethrow()
        end
    end
end

function init_pyshow()
    pyshow_add_rule(pyshow_rule_mimebundle)
    pyshow_add_rule(pyshow_rule_repr)
    pyshow_add_rule(pyshow_rule_savefig)
end
