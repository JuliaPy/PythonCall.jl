# type

"""
    DateTime64(value, [unit])
    DateTime64(value, format, [unit])

Construct an `DateTime64` with the given `value` and [`unit`](@ref Unit).

The value can be:
- An `Integer`, in which case the `unit` is required.
- A `Dates.Date` or `Dates.DateTime`.
- `"NaT"` or `"NaN"` to make a not-a-time value.
- An `AbstractString`, which is parsed the same as `Dates.DateTime`, with an optional
  `format` string.
"""
struct DateTime64 <: AbstractDateTime64
    value::Int64
    unit::Tuple{Unit,Cint}
    function DateTime64(value::Integer, unit::UnitArg)
        new(Int(value), unitpair(unit))
    end
end

# accessors

function Dates.value(d::DateTime64)
    d.value
end

function unitpair(d::DateTime64)
    d.unit
end


# constructors

function DateTime64(d::AbstractDateTime64, unit::UnitArg = defaultunit(d))
    unit = unitpair(unit)
    if unit == unitpair(d)
        DateTime64(value(d), unit)
    elseif isnan(d)
        DateTime64(NAT, unit)
    else
        error(
            "not implemented: changing units: $(unitparam(unitpair(d))) to $(unitparam(unit))",
        )
    end
end

function DateTime64(d::AbstractString, unit::UnitArg = defaultunit(d))
    unit = unitpair(unit)
    if d in NAT_STRINGS
        DateTime64(NAT, unit)
    else
        DateTime64(Dates.DateTime(d), unit)
    end
end

function DateTime64(d::AbstractString, f::Dates.DateFormat, unit::UnitArg = defaultunit(d))
    unit = unitpair(unit)
    if d in NAT_STRINGS
        DateTime64(NAT, unit)
    else
        DateTime64(Dates.DateTime(d, f), unit)
    end
end

function DateTime64(d::Dates.DateTime, unit::UnitArg = defaultunit(d))
    u, m = unit = unitpair(unit)
    if u ≤ DAYS
        return DateTime64(Dates.Date(d), unit)
    end
    v = value((d - Dates.DateTime(1970))::Dates.Millisecond)
    if u == HOURS
        m = mul(m, 1000 * 60 * 60)
    elseif u == MINUTES
        m = mul(m, 1000 * 60)
    elseif u == SECONDS
        m = mul(m, 1000)
    elseif u == MILLISECONDS
        # nothing
    elseif u == MICROSECONDS
        v = mul(v, 1000)
    elseif u == NANOSECONDS
        v = mul(v, 1000_000)
    elseif u == PICOSECONDS
        v = mul(v, 1000_000_000)
    elseif u == FEMTOSECONDS
        v = mul(v, 1000_000_000_000)
    elseif u == ATTOSECONDS
        v = mul(v, 1000_000_000_000_000)
    else
        error("unknown unit: $u")
    end
    v = fld(v, m)
    DateTime64(v, unit)
end


function DateTime64(d::Dates.Date, unit::UnitArg = defaultunit(d))
    u, m = unit = unitpair(unit)
    if u == YEARS
        v = Dates.year(d) - 1970
    elseif u == MONTHS
        v = 12 * (Dates.year(d) - 1970) + (Dates.month(d) - 1)
    else
        v = value((d - Dates.Date(1970))::Dates.Day)
        if u == WEEKS
            m = mul(m, 7)
        elseif u == DAYS
            # nothing
        elseif u == HOURS
            v = mul(v, 24)
        elseif u == MINUTES
            v = mul(v, 24 * 60)
        elseif u == SECONDS
            v = mul(v, 24 * 60 * 60)
        elseif u == MILLISECONDS
            v = mul(v, 24 * 60 * 60 * 1000)
        elseif u == MICROSECONDS
            v = mul(v, 24 * 60 * 60 * 1000_000)
        elseif u == NANOSECONDS
            v = mul(v, 24 * 60 * 60 * 1000_000_000)
        elseif u == PICOSECONDS
            v = mul(v, 24 * 60 * 60 * 1000_000_000_000)
        elseif u == FEMTOSECONDS
            throw(OverflowError(""))
        elseif u == ATTOSECONDS
            throw(OverflowError(""))
        else
            error("unknown unit: $u")
        end
    end
    v = fld(v, m)
    DateTime64(v, unit)
end


# show

function Base.show(io::IO, d::DateTime64)
    if get(io, :typeinfo, Any) == typeof(d)
        showvalue(io, d)
    else
        show(io, typeof(d))
        print(io, "(")
        showvalue(io, d)
        print(io, ", ")
        show(io, unitparam(unitpair(d)))
        print(io, ")")
    end
    nothing
end
