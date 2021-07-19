function Serialization.serialize(s::AbstractSerializer, x::Py)
    Serialization.serialize_type(s, Py, false)
    b = pyimport("pickle").dumps(x)
    Serialization.serialize(s, pybytes_asvector(b))
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{Py})
    pyimport("pickle").loads(pybytes(deserialize(s)))
end
