# type

"""
    InlineTimeDelta64{unit}(value)
    InlineTimeDelta64(value, [unit])

Construct an `InlineTimeDelta64` with the given `value` and [`unit`](@ref Unit).

The unit is part of the type, so an instance just consists of one `Int64` for the value.

The value can be:
- An `Integer`, in which case the `unit` is required.
- `"NaT"` or `"NaN"` to make a not-a-time value.
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

unitpair(::InlineTimeDelta64{U}) where {U} = unitpair(U)
unitpair(::Type{InlineTimeDelta64{U}}) where {U} = unitpair(U)

# constructors

function InlineTimeDelta64{U}(v::AbstractTimeDelta64) where {U}
    InlineTimeDelta64{U}(value(TimeDelta64(v, U)))
end

function InlineTimeDelta64{U}(v::AbstractString) where {U}
    InlineTimeDelta64{U}(value(TimeDelta64(v, U)))
end

function InlineTimeDelta64{U}(v::DatesPeriod) where {U}
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

function InlineTimeDelta64(v::DatesPeriod, u::UnitArg = defaultunit(v))
    InlineTimeDelta64{unitparam(u)}(v)
end

function InlineTimeDelta64(v::Integer, u::UnitArg)
    InlineTimeDelta64{unitparam(u)}(v)
end

# convert

Base.convert(::Type{InlineTimeDelta64}, p::InlineTimeDelta64) = p
Base.convert(::Type{InlineTimeDelta64}, p::Union{AbstractTimeDelta64,DatesPeriod}) =
    InlineTimeDelta64(p)
Base.convert(::Type{InlineTimeDelta64{U}}, p::InlineTimeDelta64{U}) where {U} = p
Base.convert(
    ::Type{InlineTimeDelta64{U}},
    p::Union{AbstractTimeDelta64,DatesPeriod},
) where {U} = InlineTimeDelta64{U}(p)

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

Base.show(io::IO, ::MIME"text/plain", d::InlineTimeDelta64) = show(io, d)
