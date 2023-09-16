### Py
#
# We use pickle to serialise Python objects to bytes.

function serialize_py(s, x::Py)
    if pyisnull(x)
        serialize(s, nothing)
    else
        b = pyimport("pickle").dumps(x)
        serialize(s, pybytes_asvector(b))
    end
end

function deserialize_py(s)
    v = deserialize(s)
    if v === nothing
        pynew()
    else
        pyimport("pickle").loads(pybytes(v))
    end
end

function Serialization.serialize(s::AbstractSerializer, x::Py)
    Serialization.serialize_type(s, Py, false)
    serialize_py(s, x)
end

Serialization.deserialize(s::AbstractSerializer, ::Type{Py}) = deserialize_py(s)

### PyException
#
# Traceback objects are not serialisable, but Exception objects are (because pickle just
# ignores the __traceback__ attribute), so we serialise a PyException by just serialising
# its (normalised) `v` field, from which we can recover the type and traceback.
#
# This means the user can install something like "tblib" to enable pickling of tracebacks
# and for free this enables serializing PyException including the traceback.

function Serialization.serialize(s::AbstractSerializer, x::PyException)
    Serialization.serialize_type(s, PyException, false)
    serialize_py(s, x.v)
end

Serialization.deserialize(s::AbstractSerializer, ::Type{PyException}) = PyException(deserialize_py(s))
