abstract type AbstractTimeDelta64 <: Dates.Period end

Dates.value(d::AbstractTimeDelta64) = d.value

function Base.isnan(d::AbstractTimeDelta64)
    value(d) == typemin(Int64)
end
