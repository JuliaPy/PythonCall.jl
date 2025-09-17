# type

"""
    InlineDateTime64{unit}(value)
    InlineDateTime64{unit}(value, format)
    InlineDateTime64(value, [unit])
    InlineDateTime64(value, format, [unit])

Construct an `InlineDateTime64` with the given `value` and [`unit`](@ref Unit).

The unit is part of the type, so an instance just consists of one `Int64` for the value.

The `value` can be:
- An `Integer`, in which case the `unit` is required.
- A `Dates.Date` or `Dates.DateTime`.
- `"NaT"` or `"NaN"` to make a not-a-time value.
- An `AbstractString`, which is parsed the same as `Dates.DateTime`, with an optional
  `format` string.
"""
struct InlineDateTime64{U} <: AbstractDateTime64
    value::Int64
    function InlineDateTime64{U}(value::Int64) where {U}
        U isa Unit ||
            U isa Tuple{Unit,Int} ||
            error("U must be a Unit or a Tuple{Unit,Int}")
        new{U}(value)
    end
end

# accessors

function Dates.value(d::InlineDateTime64)
    d.value
end

function unitpair(::InlineDateTime64{U}) where {U}
    unitpair(U)
end

# constructors

function InlineDateTime64{U}(v::AbstractDateTime64) where {U}
    InlineDateTime64{U}(value(DateTime64(v, U)))
end

function InlineDateTime64{U}(v::AbstractString) where {U}
    InlineDateTime64{U}(value(DateTime64(v, U)))
end

function InlineDateTime64{U}(v::Dates.DateTime) where {U}
    InlineDateTime64{U}(value(DateTime64(v, U)))
end

function InlineDateTime64{U}(v::Dates.Date) where {U}
    InlineDateTime64{U}(value(DateTime64(v, U)))
end

function InlineDateTime64{U}(v::Integer) where {U}
    InlineDateTime64{U}(convert(Int, v))
end

function InlineDateTime64{U}(
    v::AbstractString,
    f::Union{AbstractString,Dates.DateFormat},
) where {U}
    InlineDateTime64{U}(value(DateTime64(v, f, U)))
end

function InlineDateTime64(v::AbstractDateTime64, u::UnitArg = defaultunit(v))
    InlineDateTime64{unitparam(u)}(v)
end

function InlineDateTime64(v::AbstractString, u::UnitArg = defaultunit(v))
    InlineDateTime64{unitparam(u)}(v)
end

function InlineDateTime64(v::Dates.DateTime, u::UnitArg = defaultunit(v))
    InlineDateTime64{unitparam(u)}(v)
end

function InlineDateTime64(v::Dates.Date, u::UnitArg = defaultunit(v))
    InlineDateTime64{unitparam(u)}(v)
end

function InlineDateTime64(v::Integer, u::UnitArg)
    InlineDateTime64{unitparam(u)}(v)
end

function InlineDateTime64(
    v::AbstractString,
    f::Union{AbstractString,Dates.DateFormat},
    u::UnitArg = defaultunit(v),
)
    InlineTimeDelta64{unitparam(u)}(v, f)
end

# convert

Base.convert(::Type{InlineDateTime64}, x::InlineDateTime64) = x
Base.convert(::Type{InlineDateTime64}, x::Union{AbstractDateTime64,DatesInstant}) =
    InlineDateTime64(x)
Base.convert(::Type{InlineDateTime64{U}}, x::InlineDateTime64{U}) where {U} = x
Base.convert(
    ::Type{InlineDateTime64{U}},
    x::Union{AbstractDateTime64,DatesInstant},
) where {U} = InlineDateTime64{U}(x)

# show

function Base.show(io::IO, d::InlineDateTime64)
    if get(io, :typeinfo, Any) == typeof(d)
        showvalue(io, d)
    else
        show(io, typeof(d))
        print(io, "(")
        showvalue(io, d)
        print(io, ")")
    end
    nothing
end
