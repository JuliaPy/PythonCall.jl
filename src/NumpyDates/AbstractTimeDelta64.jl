"""
    abstract type AbstractTimeDelta64 <: Dates.Period

Supertype for [`TimeDelta64`](@ref) and [`InlineTimeDelta64`](@ref).
"""
abstract type AbstractTimeDelta64 <: Dates.Period end

function Base.isnan(d::AbstractTimeDelta64)
    value(d) == NAT
end

function showvalue(io::IO, d::AbstractTimeDelta64)
    u, m = unitpair(d)
    if isnan(d)
        show(io, "NaT")
    else
        show(io, value(d))
    end
    nothing
end

function defaultunit(d::AbstractTimeDelta64)
    unitpair(d)
end
