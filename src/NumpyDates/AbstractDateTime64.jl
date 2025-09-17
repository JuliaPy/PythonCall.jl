"""
    abstract type AbstractDateTime64 <: Dates.TimeType

Supertype for [`DateTime64`](@ref) and [`InlineDateTime64`](@ref).
"""
abstract type AbstractDateTime64 <: Dates.TimeType end

function Dates.DateTime(d::AbstractDateTime64)
    isnan(d) && error("Cannot convert NaT to DateTime")
    v = value(d)
    u, _ = unit = unitpair(d)
    b = Dates.DateTime(1970)
    if u > MONTHS
        v, _ = rescale(v, unit, MILLISECONDS)
        b + Dates.Millisecond(v)
    else
        v, _ = rescale(v, unit, MONTHS)
        b + Dates.Month(v)
    end
end

function Dates.Date(d::AbstractDateTime64)
    isnan(d) && error("Cannot convert NaT to Date")
    Dates.Date(Dates.DateTime(d))
end

Base.convert(::Type{Dates.DateTime}, d::AbstractDateTime64) = Dates.DateTime(d)
Base.convert(::Type{Dates.Date}, d::AbstractDateTime64) = Dates.Date(d)

function Base.isnan(d::AbstractDateTime64)
    value(d) == typemin(Int64)
end

function showvalue(io::IO, d::AbstractDateTime64)
    u, m = unit = unitpair(d)
    if isnan(d)
        show(io, "NaT")
    elseif u ≤ DAYS
        d2 = Dates.Date(d)
        if value(DateTime64(d2, unit)) == value(d)
            show(io, string(d2))
        else
            show(io, value(d))
        end
    else
        d2 = Dates.DateTime(d)
        if value(DateTime64(d2, unit)) == value(d)
            show(io, string(d2))
        else
            show(io, value(d))
        end
    end
    nothing
end

function defaultunit(d::AbstractDateTime64)
    unitpair(d)
end
