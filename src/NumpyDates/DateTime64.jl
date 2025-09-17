# type

"""
    DateTime64(value, [unit])
    DateTime64(value, format, [unit])

Construct an `DateTime64` with the given `value` and [`unit`](@ref Unit).

The unit is stored as a run-time value. If the units in your code are known, using
[`InlineDateTime64{unit}`](@ref InlineDateTime64) may be preferable. The memory layout
is the same as for a `numpy.datetime64`.

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

Dates.value(d::DateTime64) = d.value

unitpair(d::DateTime64) = d.unit

# constructors

function DateTime64(d::AbstractDateTime64, unit::UnitArg = defaultunit(d))
    unit = unitpair(unit)
    if unit == unitpair(d)
        DateTime64(value(d), unit)
    elseif isnan(d)
        DateTime64(NAT, unit)
    else
        v, _ = rescale(value(d), unitpair(d), unit)
        DateTime64(v, unit)
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
    u, _ = unit = unitpair(unit)
    if u == YEARS
        v_yr = sub(Dates.year(d), 1970)
        v, _ = rescale(v_yr, YEARS, unit)
    elseif u == MONTHS
        yr, mn = Dates.yearmonth(d)
        v_mn = add(mul(12, sub(yr, 1970)), sub(mn, 1))
        v, _ = rescale(v_mn, MONTHS, unit)
    else
        v_ms = d - Dates.DateTime(1970)
        v, _ = rescale(value(v_ms), Unit(v_ms), unit)
    end
    DateTime64(v, unit)
end


function DateTime64(d::Dates.Date, unit::UnitArg = defaultunit(d))
    DateTime64(Dates.DateTime(d), unit)
end

# convert

Base.convert(::Type{DateTime64}, x::DateTime64) = x
Base.convert(::Type{DateTime64}, x::AbstractDateTime64) = DateTime64(x)
Base.convert(::Type{DateTime64}, x::DatesInstant) = DateTime64(x)

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
