struct InlineTimeDelta64{U} <: AbstractTimeDelta64
    value::Int64
end

function unit(::InlineTimeDelta64{U}) where {U}
    _unit(U)
end
