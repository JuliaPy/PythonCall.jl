# type

"""
    TimeDelta64(value, [unit])

Construct a `TimeDelta64` with the given value and unit.
"""
struct TimeDelta64 <: AbstractTimeDelta64
    value::Int64
    unit::Tuple{Unit,Cint}
    function TimeDelta64(value::Integer, unit::UnitArg)
        new(Int(value), unitpair(unit))
    end
end

# accessors

function Dates.value(d::TimeDelta64)
    d.value
end

function unitpair(d::TimeDelta64)
    d.unit
end

# constructors

# (outer value/unit constructor is unnecessary; inner constructor handles UnitArg)

function TimeDelta64(d::AbstractTimeDelta64, unit::UnitArg = defaultunit(d))
    unit = unitpair(unit)
    if unit == unitpair(d)
        TimeDelta64(value(d), unit)
    elseif isnan(d)
        TimeDelta64(NAT, unit)
    else
        error(
            "not implemented: changing units: $(unitparam(unitpair(d))) to $(unitparam(unit))",
        )
    end
end

function TimeDelta64(s::AbstractString, unit::UnitArg = defaultunit(s))
    unit = unitpair(unit)
    if s in NAT_STRINGS
        TimeDelta64(NAT, unit)
    else
        error(
            "Cannot construct TimeDelta64 from string '$s'. Only NaT variants are supported.",
        )
    end
end

# Convert Dates.Period to TimeDelta64
function TimeDelta64(p::Dates.Period, unit::UnitArg = defaultunit(p))
    u, m = unitpair(unit)
    if u == YEARS
        if p isa Dates.Year
            v = value(p)
            return TimeDelta64(v ÷ m, unit)
        else
            error("cannot convert $(typeof(p)) to years")
        end
    elseif u == MONTHS
        if p isa Dates.Month
            v = value(p)
            return TimeDelta64(v ÷ m, unit)
        elseif p isa Dates.Year
            v = mul(value(p), 12)
            return TimeDelta64(v ÷ m, unit)
        else
            error("cannot convert $(typeof(p)) to months")
        end
    elseif u == PICOSECONDS || u == FEMTOSECONDS || u == ATTOSECONDS
        # sub-nanosecond units: expand from ns
        ns = _period_to_ns(p)
        scale = u == PICOSECONDS ? 1_000 : u == FEMTOSECONDS ? 1_000_000 : 1_000_000_000
        ns_scaled = mul(ns, scale)
        return TimeDelta64(ns_scaled ÷ m, unit)
    else
        # weeks..nanoseconds: convert via nanoseconds
        ns = _period_to_ns(p)
        unit_ns = _unit_to_ns(u)
        denom = mul(unit_ns, Int64(m))
        return TimeDelta64(ns ÷ denom, unit)
    end
end

# helpers

# number of nanoseconds per unit (except sub-ns which are handled separately)
function _unit_to_ns(u::Unit)::Int64
    if u == WEEKS
        mul(mul(mul(mul(Int64(7), Int64(24)), Int64(60)), Int64(60)), Int64(1_000_000_000))
    elseif u == DAYS
        mul(mul(mul(Int64(24), Int64(60)), Int64(60)), Int64(1_000_000_000))
    elseif u == HOURS
        mul(mul(Int64(60), Int64(60)), Int64(1_000_000_000))
    elseif u == MINUTES
        mul(Int64(60), Int64(1_000_000_000))
    elseif u == SECONDS
        Int64(1_000_000_000)
    elseif u == MILLISECONDS
        Int64(1_000_000)
    elseif u == MICROSECONDS
        Int64(1_000)
    elseif u == NANOSECONDS
        Int64(1)
    else
        error("Unsupported or sub-nanosecond unit for ns mapping: $u")
    end
end

# convert a Dates.Period into total nanoseconds (disallow calendar Year/Month here)
function _period_to_ns(p::Dates.Period)::Int64
    if p isa Dates.Week
        mul(value(p), _unit_to_ns(WEEKS))
    elseif p isa Dates.Day
        mul(value(p), _unit_to_ns(DAYS))
    elseif p isa Dates.Hour
        mul(value(p), _unit_to_ns(HOURS))
    elseif p isa Dates.Minute
        mul(value(p), _unit_to_ns(MINUTES))
    elseif p isa Dates.Second
        mul(value(p), _unit_to_ns(SECONDS))
    elseif p isa Dates.Millisecond
        mul(value(p), _unit_to_ns(MILLISECONDS))
    elseif p isa Dates.Microsecond
        mul(value(p), _unit_to_ns(MICROSECONDS))
    elseif p isa Dates.Nanosecond
        value(p)
    elseif p isa Dates.Month || p isa Dates.Year
        error("cannot convert $(typeof(p)) to time-based units")
    else
        error("unsupported period type: $(typeof(p))")
    end
end

# show

function Base.show(io::IO, d::TimeDelta64)
    if get(io, :typeinfo, Any) == typeof(d)
        showvalue(io, d)
    else
        show(io, typeof(d))
        print(io, "(")
        showvalue(io, d)
        print(io, ", ", unitparam(unitpair(d)), ")")
    end
    nothing
end
