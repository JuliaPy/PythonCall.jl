# type

"""
    InlineTimeDelta64{unit}(value)
    InlineTimeDelta64(value, [unit])

Construct an `InlineTimeDelta64` with the given value and unit.
"""
struct InlineTimeDelta64{U} <: AbstractTimeDelta64
    value::Int64
    function InlineTimeDelta64{U}(value::Int64) where {U}
        U isa Unit ||
            U isa Tuple{Unit,Int} ||
            error("U must be a Unit or a Tuple{Unit,Int}")
        new{U}(value)
    end
end

# accessors

function Dates.value(d::InlineTimeDelta64)
    d.value
end

function unitpair(::InlineTimeDelta64{U}) where {U}
    unitpair(U)
end

# constructors

function InlineTimeDelta64{U}(v::AbstractTimeDelta64) where {U}
    InlineTimeDelta64{U}(value(TimeDelta64(v, U)))
end

function InlineTimeDelta64{U}(v::AbstractString) where {U}
    InlineTimeDelta64{U}(value(TimeDelta64(v, U)))
end

function InlineTimeDelta64{U}(v::Dates.Period) where {U}
    InlineTimeDelta64{U}(value(TimeDelta64(v, U)))
end

function InlineTimeDelta64{U}(v::Integer) where {U}
    InlineTimeDelta64{U}(convert(Int, v))
end

function InlineTimeDelta64(v::AbstractTimeDelta64, u::UnitArg = defaultunit(v))
    InlineTimeDelta64{unitparam(u)}(v)
end

function InlineTimeDelta64(v::AbstractString, u::UnitArg = defaultunit(v))
    InlineTimeDelta64{unitparam(u)}(v)
end

function InlineTimeDelta64(v::Dates.Period, u::UnitArg = defaultunit(v))
    InlineTimeDelta64{unitparam(u)}(v)
end

function InlineTimeDelta64(v::Integer, u::UnitArg)
    InlineTimeDelta64{unitparam(u)}(v)
end

# show

function Base.show(io::IO, d::InlineTimeDelta64)
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
