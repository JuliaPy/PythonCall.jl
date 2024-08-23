module Utils

function explode_union(T)
    @nospecialize T

    # unpack unionall
    S = T
    vars = []
    while S isa UnionAll
        pushfirst!(vars, S.var)
        S = S.body
    end

    if S isa Union
        Us = Any[explode_union(S.a)..., explode_union(S.b)...]
        Any[foldl((body, var) -> UnionAll(var, body), vars, init = U) for U in Us]
    elseif S == Union{}
        Any[]
    else
        Any[T]
    end
end

"""
    pointer_from_obj(x)

Returns `(p, c)` where `Base.pointer_from_objref(p) === x`.

The pointer remains valid provided the object `c` is not garbage collected.
"""
function pointer_from_obj(o::T) where {T}
    if T.mutable
        c = o
        p = Base.pointer_from_objref(o)
    else
        c = Ref{Any}(o)
        p = unsafe_load(Ptr{Ptr{Cvoid}}(Base.pointer_from_objref(c)))
    end
    p, c
end

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
        mimetype = _unwrap_unionall(meth.sig).parameters[3]
        mimetype isa DataType || continue
        mimetype <: MIME || continue
        mime = string(mimetype.parameters[1])
        push!(mimes, mime)
    end
    return mimes
end

@generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)

@generated _type_ub(::Type{T}) where {T} = begin
    S = T
    while S isa UnionAll
        S = S{S.var.ub}
    end
    S
end

@generated _type_lb(::Type{T}) where {T} = begin
    R = _unwrap_unionall(T)
    if R isa DataType
        S = T
        while S isa UnionAll
            S = S{S.var in R.parameters ? S.var.lb : S.var.ub}
        end
        S
    else
        _type_ub(T)
    end
end

@generated function _unwrap_unionall(::Type{T}) where {T}
    R = T
    while R isa UnionAll
        R = R.body
    end
    R
end

@generated _promote_type_bounded(::Type{S}, ::Type{T}, ::Type{B}) where {S,T,B} = begin
    S <: B || error("require S <: B")
    T <: B || error("require T <: B")
    if B isa Union
        return Union{
            _promote_type_bounded(typeintersect(S, B.a), typeintersect(T, B.a), B.a),
            _promote_type_bounded(typeintersect(S, B.b), typeintersect(T, B.b), B.b),
        }
    else
        R = promote_type(S, T)
        if R <: B
            return R
        else
            R = typeintersect(typejoin(S, T), B)
            if R <: B
                return R
            else
                return B
            end
        end
    end
end

@generated _promote_type_bounded(
    ::Type{T1},
    ::Type{T2},
    ::Type{T3},
    ::Type{B},
) where {T1,T2,T3,B} = _promote_type_bounded(_promote_type_bounded(T1, T2, B), T3, B)

# TODO: what is the best way?
ismutablearray(x::Array) = true
ismutablearray(x::AbstractArray) = begin
    p = parent(x)
    p === x ? false : ismutablearray(p)
end

islittleendian() =
    Base.ENDIAN_BOM == 0x04030201 ? true : Base.ENDIAN_BOM == 0x01020304 ? false : error()

isflagset(flags, mask) = (flags & mask) == mask

size_to_fstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
    isempty(sz) ? () : (elsz, size_to_fstrides(elsz * sz[1], sz[2:end])...)

size_to_cstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
    isempty(sz) ? () : (size_to_cstrides(elsz * sz[end], sz[1:end-1])..., elsz)

struct StaticString{T,N} <: AbstractString
    codeunits::NTuple{N,T}
    StaticString{T,N}(codeunits::NTuple{N,T}) where {T,N} = new{T,N}(codeunits)
end

function Base.String(x::StaticString{T,N}) where {T,N}
    ts = x.codeunits
    n = N
    while n > 0 && iszero(ts[n])
        n -= 1
    end
    cs = T[ts[i] for i = 1:n]
    transcode(String, cs)
end

function Base.convert(::Type{StaticString{T,N}}, x::AbstractString) where {T,N}
    ts = transcode(T, convert(String, x))
    n = length(ts)
    n > N && throw(InexactError(:convert, StaticString{T,N}, x))
    n > 0 && iszero(ts[n]) && throw(InexactError(:convert, StaticString{T,N}, x))
    z = zero(T)
    cs = ntuple(i -> i > n ? z : @inbounds(ts[i]), N)
    StaticString{T,N}(cs)
end

StaticString{T,N}(x::AbstractString) where {T,N} = convert(StaticString{T,N}, x)

Base.ncodeunits(x::StaticString{T,N}) where {T,N} = N

Base.codeunit(x::StaticString, i::Integer) = x.codeunits[i]

Base.codeunit(x::StaticString{T}) where {T} = T

function Base.isvalid(x::StaticString{UInt8,N}, i::Int) where {N}
    if i < 1 || i > N
        return false
    end
    cs = x.codeunits
    c = @inbounds cs[i]
    if all(iszero, (cs[j] for j = i:N))
        return false
    elseif (c & 0x80) == 0x00
        return true
    elseif (c & 0x40) == 0x00
        return false
    elseif (c & 0x20) == 0x00
        return @inbounds (i ≤ N - 1) && ((cs[i+1] & 0xC0) == 0x80)
    elseif (c & 0x10) == 0x00
        return @inbounds (i ≤ N - 2) &&
                         ((cs[i+1] & 0xC0) == 0x80) &&
                         ((cs[i+2] & 0xC0) == 0x80)
    elseif (c & 0x08) == 0x00
        return @inbounds (i ≤ N - 3) &&
                         ((cs[i+1] & 0xC0) == 0x80) &&
                         ((cs[i+2] & 0xC0) == 0x80) &&
                         ((cs[i+3] & 0xC0) == 0x80)
    else
        return false
    end
    return false
end

function Base.iterate(x::StaticString{UInt8,N}, i::Int = 1) where {N}
    i > N && return
    cs = x.codeunits
    c = @inbounds cs[i]
    if all(iszero, (cs[j] for j = i:N))
        return
    elseif (c & 0x80) == 0x00
        return (reinterpret(Char, UInt32(c) << 24), i + 1)
    elseif (c & 0x40) == 0x00
        nothing
    elseif (c & 0x20) == 0x00
        if @inbounds (i ≤ N - 1) && ((cs[i+1] & 0xC0) == 0x80)
            return (
                reinterpret(Char, (UInt32(cs[i]) << 24) | (UInt32(cs[i+1]) << 16)),
                i + 2,
            )
        end
    elseif (c & 0x10) == 0x00
        if @inbounds (i ≤ N - 2) && ((cs[i+1] & 0xC0) == 0x80) && ((cs[i+2] & 0xC0) == 0x80)
            return (
                reinterpret(
                    Char,
                    (UInt32(cs[i]) << 24) |
                    (UInt32(cs[i+1]) << 16) |
                    (UInt32(cs[i+2]) << 8),
                ),
                i + 3,
            )
        end
    elseif (c & 0x08) == 0x00
        if @inbounds (i ≤ N - 3) &&
                     ((cs[i+1] & 0xC0) == 0x80) &&
                     ((cs[i+2] & 0xC0) == 0x80) &&
                     ((cs[i+3] & 0xC0) == 0x80)
            return (
                reinterpret(
                    Char,
                    (UInt32(cs[i]) << 24) |
                    (UInt32(cs[i+1]) << 16) |
                    (UInt32(cs[i+2]) << 8) |
                    UInt32(cs[i+3]),
                ),
                i + 4,
            )
        end
    end
    throw(StringIndexError(x, i))
end

function Base.isvalid(x::StaticString{UInt32,N}, i::Int) where {N}
    i < 1 && return false
    cs = x.codeunits
    return !all(iszero, (cs[j] for j = i:N))
end

function Base.iterate(x::StaticString{UInt32,N}, i::Int = 1) where {N}
    i > N && return
    cs = x.codeunits
    c = @inbounds cs[i]
    if all(iszero, (cs[j] for j = i:N))
        return
    else
        return (Char(c), i + 1)
    end
end

end
