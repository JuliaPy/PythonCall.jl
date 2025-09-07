struct TimeDelta64 <: AbstractTimeDelta64
    value::Int64
    unit_base::Unit
    unit_scale::Cint
end

function unit(d::TimeDelta64)
    (d.unit_base, d.unit_scale)
end
