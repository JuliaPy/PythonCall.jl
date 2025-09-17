"""
    abstract type AbstractTimeDelta64 <: Dates.Period

Supertype for [`TimeDelta64`](@ref) and [`InlineTimeDelta64`](@ref).
"""
abstract type AbstractTimeDelta64 <: Dates.Period end

function construct(::Type{T}, d::AbstractTimeDelta64) where {T<:DatesPeriod}
    v, r = rescale(value(d), unitpair(d), Unit(T))
    iszero(r) || throw(InexactError(nameof(T), T, d))
    T(v)
end

Dates.Year(d::AbstractTimeDelta64) = construct(Dates.Year, d)
Dates.Month(d::AbstractTimeDelta64) = construct(Dates.Month, d)
Dates.Day(d::AbstractTimeDelta64) = construct(Dates.Day, d)
Dates.Hour(d::AbstractTimeDelta64) = construct(Dates.Hour, d)
Dates.Minute(d::AbstractTimeDelta64) = construct(Dates.Minute, d)
Dates.Second(d::AbstractTimeDelta64) = construct(Dates.Second, d)
Dates.Millisecond(d::AbstractTimeDelta64) = construct(Dates.Millisecond, d)
Dates.Microsecond(d::AbstractTimeDelta64) = construct(Dates.Microsecond, d)
Dates.Nanosecond(d::AbstractTimeDelta64) = construct(Dates.Nanosecond, d)

Base.convert(::Type{Dates.Year}, d::AbstractTimeDelta64) = Dates.Year(d)
Base.convert(::Type{Dates.Month}, d::AbstractTimeDelta64) = Dates.Month(d)
Base.convert(::Type{Dates.Day}, d::AbstractTimeDelta64) = Dates.Day(d)
Base.convert(::Type{Dates.Hour}, d::AbstractTimeDelta64) = Dates.Hour(d)
Base.convert(::Type{Dates.Minute}, d::AbstractTimeDelta64) = Dates.Minute(d)
Base.convert(::Type{Dates.Second}, d::AbstractTimeDelta64) = Dates.Second(d)
Base.convert(::Type{Dates.Millisecond}, d::AbstractTimeDelta64) = Dates.Millisecond(d)
Base.convert(::Type{Dates.Microsecond}, d::AbstractTimeDelta64) = Dates.Microsecond(d)
Base.convert(::Type{Dates.Nanosecond}, d::AbstractTimeDelta64) = Dates.Nanosecond(d)

function Base.isnan(d::AbstractTimeDelta64)
    value(d) == NAT
end

function showvalue(io::IO, d::AbstractTimeDelta64)
    u, m = unitpair(d)
    if isnan(d)
        show(io, "NaT")
    else
        show(io, value(d))
    end
    nothing
end

function defaultunit(d::AbstractTimeDelta64)
    unitpair(d)
end
