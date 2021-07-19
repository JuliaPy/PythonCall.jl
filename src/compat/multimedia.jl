py_mime_reprmethod(::MIME) = nothing
py_mime_reprmethod(::MIME"text/plain") = "__repr__"
py_mime_reprmethod(::MIME"text/html") = "_repr_html_"
py_mime_reprmethod(::MIME"text/markdown") = "_repr_markdown_"
py_mime_reprmethod(::MIME"text/json") = "_repr_json_"
py_mime_reprmethod(::MIME"text/latex") = "_repr_latex_"
py_mime_reprmethod(::MIME"application/javascript") = "_repr_javascript_"
py_mime_reprmethod(::MIME"application/pdf") = "_repr_pdf_"
py_mime_reprmethod(::MIME"image/jpeg") = "_repr_jpeg_"
py_mime_reprmethod(::MIME"image/png") = "_repr_png_"
py_mime_reprmethod(::MIME"image/svg+xml") = "_repr_svg_"

py_mime_data(m::MIME, o) = begin
    o = Py(o)
    r = py_mime_reprmethod(m)
    data = nothing
    meta = nothing
    try
        x = o.__class__._repr_mimebundle_(o, include=pylist((string(m),)))
        if pyisinstance(x, pybuiltins.tuple)
            data = x[0][string(m)]
            meta = x[1].get(string(m))
        else
            data = x[m]
        end
    catch exc
        exc isa PyException || rethrow()
    end
    if data === nothing && r !== nothing
        try
            x = pygetattr(o.__class__, r)(o)
            if pyisinstance(x, pybuiltins.tuple)
                data = x[0]
                meta = x[1]
            else
                data = x
            end
        catch exc
            exc isa PyException || rethrow()
        end
    end
    data, meta
end

py_mime_showable(m::MIME, o) = begin
    data, meta = py_mime_data(m, o)
    data !== nothing
end

py_mime_show(io::IO, m::MIME, o) = begin
    data, meta = py_mime_data(m, o)
    write(io, istextmime(m) ? pystr(String, data) : pybytes(Vector{UInt8}, data))
    nothing
end
