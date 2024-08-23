### Py
#
# We use pickle to serialise Python objects to bytes.

_pickle_module() = pyimport(get(ENV, "JULIA_PYTHONCALL_PICKLE", "pickle"))

function serialize_py(s, x::Py)
    if pyisnull(x)
        serialize(s, nothing)
    else
        b = _pickle_module().dumps(x)
        serialize(s, pybytes_asvector(b))
    end
end

function deserialize_py(s)
    v = deserialize(s)
    if v === nothing
        pynew()
    else
        _pickle_module().loads(pybytes(v))
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

Serialization.deserialize(s::AbstractSerializer, ::Type{PyException}) =
    PyException(deserialize_py(s))

### PyArray
#
# This type holds a pointer and a handle (usually a python memoryview or capsule) which are
# not serializable by default, and even if they were would not be consistent after
# serializing each field independently. So we just serialize the wrapped Python object.

function Serialization.serialize(s::AbstractSerializer, x::PyArray)
    Serialization.serialize_type(s, typeof(x), false)
    serialize_py(s, x.py)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{T}) where {T<:PyArray}
    # TODO: set buffer and array args too?
    T(deserialize_py(s); copy = false)
end
