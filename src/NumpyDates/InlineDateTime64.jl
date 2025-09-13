# type

"""
    InlineDateTime64{unit}(value)
    InlineDateTime64{unit}(value, format)
    InlineDateTime64(value, [unit])
    InlineDateTime64(value, format, [unit])

Construct an `InlineDateTime64` with the given value and unit.
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

function InlineDateTime64{U}(
    v::Union{AbstractDateTime64,AbstractString,Dates.DateTime,Dates.Date},
) where {U}
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

function InlineDateTime64(
    v::Union{AbstractDateTime64,AbstractString,Dates.DateTime,Dates.Date},
    u::UnitArg = defaultunit(v),
)
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
