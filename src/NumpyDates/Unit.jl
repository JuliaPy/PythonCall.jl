"""
    @enum Unit

The possible time units for datetimes and timedeltas in this module.

Values are: `YEARS`, `MONTHS`, `WEEKS`, `DAYS`, `HOURS`, `MINUTES`, `SECONDS`,
`MILLISECONDS`, `MICROSECONDS`, `NANOSECONDS`, `PICOSECONDS`, `FEMTOSECONDS`,
`ATTOSECONDS`.

For compatibility with numpy, the types in this module also accept scaled units as a
`Tuple{Unit,Int}`. For example `(MINUTES, 15)` for units of 15 minutes. This feature is
rarely used.
"""
@enum Unit::Cint begin
    YEARS = 0
    MONTHS = 1
    WEEKS = 2
    DAYS = 4
    HOURS = 5
    MINUTES = 6
    SECONDS = 7
    MILLISECONDS = 8
    MICROSECONDS = 9
    NANOSECONDS = 10
    PICOSECONDS = 11
    FEMTOSECONDS = 12
    ATTOSECONDS = 13
    UNBOUND_UNITS = 14
end

function Unit(u::Unit)
    u
end

function Unit(u::Symbol)
    u == :Y ? YEARS :
    u == :M ? MONTHS :
    u == :W ? WEEKS :
    u == :D ? DAYS :
    u == :h ? HOURS :
    u == :m ? MINUTES :
    u == :s ? SECONDS :
    u == :ms ? MILLISECONDS :
    u == :us ? MICROSECONDS :
    u == :ns ? NANOSECONDS :
    u == :ps ? PICOSECONDS :
    u == :fs ? FEMTOSECONDS : u == :as ? ATTOSECONDS : error("invalid unit: $u")
end

Unit(::Type{Dates.Date}) = DAYS
Unit(::Type{Dates.DateTime}) = MILLISECONDS
Unit(::Type{Dates.Year}) = YEARS
Unit(::Type{Dates.Month}) = MONTHS
Unit(::Type{Dates.Week}) = WEEKS
Unit(::Type{Dates.Day}) = DAYS
Unit(::Type{Dates.Hour}) = HOURS
Unit(::Type{Dates.Minute}) = MINUTES
Unit(::Type{Dates.Second}) = SECONDS
Unit(::Type{Dates.Millisecond}) = MILLISECONDS
Unit(::Type{Dates.Microsecond}) = MICROSECONDS
Unit(::Type{Dates.Nanosecond}) = NANOSECONDS

Unit(d::DatesInstant) = Unit(typeof(d))
Unit(p::DatesPeriod) = Unit(typeof(p))

function Base.Symbol(u::Unit)
    u == NumpyDates.YEARS ? :Y :
    u == NumpyDates.MONTHS ? :M :
    u == NumpyDates.WEEKS ? :W :
    u == NumpyDates.DAYS ? :D :
    u == NumpyDates.HOURS ? :h :
    u == NumpyDates.MINUTES ? :m :
    u == NumpyDates.SECONDS ? :s :
    u == NumpyDates.MILLISECONDS ? :ms :
    u == NumpyDates.MICROSECONDS ? :us :
    u == NumpyDates.NANOSECONDS ? :ns :
    u == NumpyDates.PICOSECONDS ? :ps :
    u == NumpyDates.FEMTOSECONDS ? :fs :
    u == NumpyDates.ATTOSECONDS ? :as : error("invalid unit: $u")
end

const UnitArg = Union{Unit,Symbol,Tuple{Unit,Integer},Tuple{Symbol,Integer}}

const UnitPair = Tuple{Unit,Integer}

function unitpair(u::UnitArg)
    u, m = u isa Tuple ? u : (u, 1)
    (Unit(u), m)
end

function unitparam(u::UnitArg)
    u, m = unitpair(u)
    m == 1 ? u : (u, Int(m))
end

defaultunit(::Any) = NANOSECONDS
defaultunit(d::DatesInstant) = Unit(d)
defaultunit(p::DatesPeriod) = Unit(p)

"""
    unitscale(u0::Unit, u1::Unit)

Returns `(multiplier, divisor)` to go from a value in units `u0` to units `u1`.

For example `unitscale(YEARS, MONTHS)` returns `(12, 1)` because 1 year is 12 months.
"""
function unitscale(u0::Unit, u1::Unit)
    m::Int64 = 1
    d::Int64 = 1
    c(n) = n isa Int128 ? Int64(0) : Int64(n)
    if u0 == u1
        # ok
    elseif u0 == YEARS
        if u1 == YEARS
            # ok
        elseif u1 == MONTHS
            m = c(12)
        else
            m = 0
        end
    elseif u0 == MONTHS
        if u1 == YEARS
            d = c(12)
        elseif u1 == MONTHS
            # ok
        else
            m = 0
        end
    elseif u0 == WEEKS
        if u1 == WEEKS
            # ok
        elseif u1 == DAYS
            m = c(7)
        elseif u1 == HOURS
            m = c(168)
        elseif u1 == MINUTES
            m = c(10_080)
        elseif u1 == SECONDS
            m = c(604_800)
        elseif u1 == MILLISECONDS
            m = c(604_800_000)
        elseif u1 == MICROSECONDS
            m = c(604_800_000_000)
        elseif u1 == NANOSECONDS
            m = c(604_800_000_000_000)
        elseif u1 == PICOSECONDS
            m = c(604_800_000_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(604_800_000_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(604_800_000_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == DAYS
        if u1 == WEEKS
            d = c(7)
        elseif u1 == DAYS
            # ok
        elseif u1 == HOURS
            m = c(24)
        elseif u1 == MINUTES
            m = c(1_440)
        elseif u1 == SECONDS
            m = c(86_400)
        elseif u1 == MILLISECONDS
            m = c(86_400_000)
        elseif u1 == MICROSECONDS
            m = c(86_400_000_000)
        elseif u1 == NANOSECONDS
            m = c(86_400_000_000_000)
        elseif u1 == PICOSECONDS
            m = c(86_400_000_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(86_400_000_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(86_400_000_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == HOURS
        if u1 == WEEKS
            d = c(168)
        elseif u1 == DAYS
            d = c(24)
        elseif u1 == HOURS
            # ok
        elseif u1 == MINUTES
            m = c(60)
        elseif u1 == SECONDS
            m = c(3_600)
        elseif u1 == MILLISECONDS
            m = c(3_600_000)
        elseif u1 == MICROSECONDS
            m = c(3_600_000_000)
        elseif u1 == NANOSECONDS
            m = c(3_600_000_000_000)
        elseif u1 == PICOSECONDS
            m = c(3_600_000_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(3_600_000_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(3_600_000_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == MINUTES
        if u1 == WEEKS
            d = c(10_080)
        elseif u1 == DAYS
            d = c(1_440)
        elseif u1 == HOURS
            d = c(60)
        elseif u1 == MINUTES
            # ok
        elseif u1 == SECONDS
            m = c(60)
        elseif u1 == MILLISECONDS
            m = c(60_000)
        elseif u1 == MICROSECONDS
            m = c(60_000_000)
        elseif u1 == NANOSECONDS
            m = c(60_000_000_000)
        elseif u1 == PICOSECONDS
            m = c(60_000_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(60_000_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(60_000_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == SECONDS
        if u1 == WEEKS
            d = c(604_800)
        elseif u1 == DAYS
            d = c(86_400)
        elseif u1 == HOURS
            d = c(3_600)
        elseif u1 == MINUTES
            d = c(60)
        elseif u1 == SECONDS
            # ok
        elseif u1 == MILLISECONDS
            m = c(1_000)
        elseif u1 == MICROSECONDS
            m = c(1_000_000)
        elseif u1 == NANOSECONDS
            m = c(1_000_000_000)
        elseif u1 == PICOSECONDS
            m = c(1_000_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(1_000_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(1_000_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == MILLISECONDS
        if u1 == WEEKS
            d = c(604_800_000)
        elseif u1 == DAYS
            d = c(86_400_000)
        elseif u1 == HOURS
            d = c(3_600_000)
        elseif u1 == MINUTES
            d = c(60_000)
        elseif u1 == SECONDS
            d = c(1_000)
        elseif u1 == MILLISECONDS
            # ok
        elseif u1 == MICROSECONDS
            m = c(1_000)
        elseif u1 == NANOSECONDS
            m = c(1_000_000)
        elseif u1 == PICOSECONDS
            m = c(1_000_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(1_000_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(1_000_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == MICROSECONDS
        if u1 == WEEKS
            d = c(604_800_000_000)
        elseif u1 == DAYS
            d = c(86_400_000_000)
        elseif u1 == HOURS
            d = c(3_600_000_000)
        elseif u1 == MINUTES
            d = c(60_000_000)
        elseif u1 == SECONDS
            d = c(1_000_000)
        elseif u1 == MILLISECONDS
            d = c(1_000)
        elseif u1 == MICROSECONDS
            # ok
        elseif u1 == NANOSECONDS
            m = c(1_000)
        elseif u1 == PICOSECONDS
            m = c(1_000_000)
        elseif u1 == FEMTOSECONDS
            m = c(1_000_000_000)
        elseif u1 == ATTOSECONDS
            m = c(1_000_000_000_000)
        else
            m = 0
        end
    elseif u0 == NANOSECONDS
        if u1 == WEEKS
            d = c(604_800_000_000_000)
        elseif u1 == DAYS
            d = c(86_400_000_000_000)
        elseif u1 == HOURS
            d = c(3_600_000_000_000)
        elseif u1 == MINUTES
            d = c(60_000_000_000)
        elseif u1 == SECONDS
            d = c(1_000_000_000)
        elseif u1 == MILLISECONDS
            d = c(1_000_000)
        elseif u1 == MICROSECONDS
            d = c(1_000)
        elseif u1 == NANOSECONDS
            # ok
        elseif u1 == PICOSECONDS
            m = c(1_000)
        elseif u1 == FEMTOSECONDS
            m = c(1_000_000)
        elseif u1 == ATTOSECONDS
            m = c(1_000_000_000)
        else
            m = 0
        end
    elseif u0 == PICOSECONDS
        if u1 == WEEKS
            d = c(604_800_000_000_000_000)
        elseif u1 == DAYS
            d = c(86_400_000_000_000_000)
        elseif u1 == HOURS
            d = c(3_600_000_000_000_000)
        elseif u1 == MINUTES
            d = c(60_000_000_000_000)
        elseif u1 == SECONDS
            d = c(1_000_000_000_000)
        elseif u1 == MILLISECONDS
            d = c(1_000_000_000)
        elseif u1 == MICROSECONDS
            d = c(1_000_000)
        elseif u1 == NANOSECONDS
            d = c(1_000)
        elseif u1 == PICOSECONDS
            # ok
        elseif u1 == FEMTOSECONDS
            m = c(1_000)
        elseif u1 == ATTOSECONDS
            m = c(1_000_000)
        else
            m = 0
        end
    elseif u0 == FEMTOSECONDS
        if u1 == WEEKS
            d = c(604_800_000_000_000_000_000)
        elseif u1 == DAYS
            d = c(86_400_000_000_000_000_000)
        elseif u1 == HOURS
            d = c(3_600_000_000_000_000_000)
        elseif u1 == MINUTES
            d = c(60_000_000_000_000_000)
        elseif u1 == SECONDS
            d = c(1_000_000_000_000_000)
        elseif u1 == MILLISECONDS
            d = c(1_000_000_000_000)
        elseif u1 == MICROSECONDS
            d = c(1_000_000_000)
        elseif u1 == NANOSECONDS
            d = c(1_000_000)
        elseif u1 == PICOSECONDS
            d = c(1_000)
        elseif u1 == FEMTOSECONDS
            # ok
        elseif u1 == ATTOSECONDS
            m = c(1_000)
        else
            m = 0
        end
    elseif u0 == ATTOSECONDS
        if u1 == WEEKS
            d = c(604_800_000_000_000_000_000_000)
        elseif u1 == DAYS
            d = c(86_400_000_000_000_000_000_000)
        elseif u1 == HOURS
            d = c(3_600_000_000_000_000_000_000)
        elseif u1 == MINUTES
            d = c(60_000_000_000_000_000_000)
        elseif u1 == SECONDS
            d = c(1_000_000_000_000_000_000)
        elseif u1 == MILLISECONDS
            d = c(1_000_000_000_000_000)
        elseif u1 == MICROSECONDS
            d = c(1_000_000_000_000)
        elseif u1 == NANOSECONDS
            d = c(1_000_000_000)
        elseif u1 == PICOSECONDS
            d = c(1_000_000)
        elseif u1 == FEMTOSECONDS
            d = c(1_000)
        elseif u1 == ATTOSECONDS
            # ok
        else
            m = 0
        end
    else
        m = 0
    end
    (m == 0 || d == 0) && error("cannot convert from $u0 to $u1")
    (m, d)
end

"""
    rescale(value::Int64, from_unit, to_unit)

Given a `value` in a given `from_unit`, rescale it to be in the `to_unit`.

Returns `(to_value, remainder)` where `remainder`

For example `rescale(2, YEARS, MONTHS) == (24, 0)` because 2 years is exactly 24 months.

And `rescale(2001, MILLISECONDS, SECONDS) == (2, 1)` because 2001 milliseconds is 2
seconds and 1 millisecond.
"""
function rescale(v::Int64, from_unit, to_unit)
    iszero(v) && return (v, zero(Int64))
    u0::Unit, m0::Int64 = unitpair(from_unit)
    u1::Unit, m1::Int64 = unitpair(to_unit)
    (multiplier, divisor) = unitscale(u0, u1)
    multiplier = mul(multiplier, m0)
    divisor = mul(divisor, m1)
    fldmod(mul(v, multiplier), divisor)
end
