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
