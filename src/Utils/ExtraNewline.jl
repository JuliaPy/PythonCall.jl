"""
    ExtraNewline(x)

An object that displays the same as `x` but with an extra newline in text/plain.
"""
struct ExtraNewline{T}
    value::T
end
Base.show(io::IO, m::MIME, x::ExtraNewline) = show(io, m, x.value)
Base.show(io::IO, m::MIME"text/csv", x::ExtraNewline) = show(io, m, x.value)
Base.show(io::IO, m::MIME"text/tab-separated-values", x::ExtraNewline) =
    show(io, m, x.value)
Base.show(io::IO, m::MIME"text/plain", x::ExtraNewline) =
    (show(io, m, x.value); println(io))
Base.showable(m::MIME, x::ExtraNewline) = showable(m, x.value)
