# type

"""
    TimeDelta64(value, [unit])

Construct a `TimeDelta64` with the given `value` and [`unit`](@ref Unit).

The unit is stored as a run-time value. If the units in your code are known, using
[`InlineTimeDelta64{unit}`](@ref InlineTimeDelta64) may be preferable. The memory layout
is the same as for a `numpy.timedelta64`.

The value can be:
- An `Integer`, in which case the `unit` is required.
- `"NaT"` or `"NaN"` to make a not-a-time value.
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
        v, r = rescale(value(d), unitpair(d), unit)
        iszero(r) || throw(InexactError(:TimeDelta64, TimeDelta64, d, unit))
        TimeDelta64(v, unit)
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

function TimeDelta64(p::DatesPeriod, unit::UnitArg = defaultunit(p))
    v, r = rescale(value(p), Unit(p), unit)
    iszero(r) || throw(InexactError(:TimeDelta64, TimeDelta64, p, unit))
    TimeDelta64(v, unit)
end

# convert

Base.convert(::Type{TimeDelta64}, p::TimeDelta64) = p
Base.convert(::Type{TimeDelta64}, p::Union{AbstractTimeDelta64,DatesPeriod}) =
    TimeDelta64(p)

# show

function Base.show(io::IO, d::TimeDelta64)
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

Base.show(io::IO, ::MIME"text/plain", d::TimeDelta64) = show(io, d)
