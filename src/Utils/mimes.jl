const ALL_MIMES = [
    "text/plain",
    "text/html",
    "text/markdown",
    "text/json",
    "text/latex",
    "text/xml",
    "text/csv",
    "application/javascript",
    "application/pdf",
    "application/ogg",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "audio/aac",
    "audio/mpeg",
    "audio/ogg",
    "audio/opus",
    "audio/webm",
    "audio/wav",
    "audio/midi",
    "audio/x-midi",
    "video/mpeg",
    "video/ogg",
    "video/webm",
]

function mimes_for(x)
    @nospecialize x
    # default mimes we always try
    mimes = copy(ALL_MIMES)
    # look for mimes on show methods for this type
    for meth in methods(show, Tuple{IO,MIME,typeof(x)}).ms
        mimetype = unwrap_unionall(meth.sig).parameters[3]
        mimetype isa DataType || continue
        mimetype <: MIME || continue
        mime = string(mimetype.parameters[1])
        push!(mimes, mime)
    end
    return mimes
end
