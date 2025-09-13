abstract type AbstractDateTime64 <: Dates.TimeType end

function Dates.value(d::AbstractDateTime64)
    d.value
end

function Dates.DateTime(d::AbstractDateTime64)
    isnan(d) && error("Cannot convert NaT to DateTime")
    v = value(d)
    u, s = unitpair(d)
    v = v * s  # TODO: check overflow
    b = Dates.DateTime(1970)
    if u == YEARS
        b + Dates.Year(v)
    elseif u == MONTHS
        b + Dates.Month(v)
    elseif u == WEEKS
        b + Dates.Week(v)
    elseif u == DAYS
        b + Dates.Day(v)
    elseif u == HOURS
        b + Dates.Hour(v)
    elseif u == MINUTES
        b + Dates.Minute(v)
    elseif u == SECONDS
        b + Dates.Second(v)
    elseif u == MILLISECONDS
        b + Dates.Millisecond(v)
    elseif u == MICROSECONDS
        b + Dates.Microsecond(v)
    elseif u == NANOSECONDS
        b + Dates.Nanosecond(v)
    elseif u == PICOSECONDS
        b + Dates.Nanosecond(fld(v, 1_000))
    elseif u == FEMTOSECONDS
        b + Dates.Nanosecond(fld(v, 1_000_000))
    elseif u == ATTOSECONDS
        b + Dates.Nanosecond(fld(v, 1_000_000_000))
    else
        error("Unsupported units: $unit_base")
    end
end

function Dates.Date(d::AbstractDateTime64)
    isnan(d) && error("Cannot convert NaT to Date")
    Dates.Date(Dates.DateTime(d))
end

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
