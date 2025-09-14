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
    u == :Y ? NumpyDates.YEARS :
    u == :M ? NumpyDates.MONTHS :
    u == :W ? NumpyDates.WEEKS :
    u == :D ? NumpyDates.DAYS :
    u == :h ? NumpyDates.HOURS :
    u == :m ? NumpyDates.MINUTES :
    u == :s ? NumpyDates.SECONDS :
    u == :ms ? NumpyDates.MILLISECONDS :
    u == :us ? NumpyDates.MICROSECONDS :
    u == :ns ? NumpyDates.NANOSECONDS :
    u == :ps ? NumpyDates.PICOSECONDS :
    u == :fs ? NumpyDates.FEMTOSECONDS :
    u == :as ? NumpyDates.ATTOSECONDS : error("invalid unit: $u")
end

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

function unitpair(u::UnitArg)
    u, m = u isa Tuple ? u : (u, 1)
    (Unit(u), m)
end

function unitparam(u::UnitArg)
    u, m = unitpair(u)
    m == 1 ? u : (u, Int(m))
end

function defaultunit(::Any)
    NANOSECONDS
end

function defaultunit(::Dates.Date)
    DAYS
end

function defaultunit(::Dates.DateTime)
    MILLISECONDS
end

function defaultunit(::Dates.Year)
    YEARS
end

function defaultunit(::Dates.Month)
    MONTHS
end

function defaultunit(::Dates.Week)
    WEEKS
end

function defaultunit(::Dates.Day)
    DAYS
end

function defaultunit(::Dates.Hour)
    HOURS
end

function defaultunit(::Dates.Minute)
    MINUTES
end

function defaultunit(::Dates.Second)
    SECONDS
end

function defaultunit(::Dates.Millisecond)
    MILLISECONDS
end

function defaultunit(::Dates.Microsecond)
    MICROSECONDS
end

function defaultunit(::Dates.Nanosecond)
    NANOSECONDS
end
