module Serialization

using ...PythonCall
using ...Core: pyisnull, pybytes_asvector, pynew

using Serialization: Serialization as S

_pickle_module() = pyimport(get(ENV, "JULIA_PYTHONCALL_PICKLE", "pickle"))

function serialize_py(s, x::Py)
    if pyisnull(x)
        S.serialize(s, nothing)
    else
        b = _pickle_module().dumps(x)
        S.serialize(s, pybytes_asvector(b))
    end
end

function deserialize_py(s)
    v = S.deserialize(s)
    if v === nothing
        pynew()
    else
        _pickle_module().loads(pybytes(v))
    end
end

function S.serialize(s::S.AbstractSerializer, x::Py)
    S.serialize_type(s, Py, false)
    serialize_py(s, x)
end

S.deserialize(s::S.AbstractSerializer, ::Type{Py}) = deserialize_py(s)

### PyException
#
# Traceback objects are not serialisable, but Exception objects are (because pickle just
# ignores the __traceback__ attribute), so we serialise a PyException by just serialising
# its (normalised) `v` field, from which we can recover the type and traceback.
#
# This means the user can install something like "tblib" to enable pickling of tracebacks
# and for free this enables serializing PyException including the traceback.

function S.serialize(s::S.AbstractSerializer, x::PyException)
    S.serialize_type(s, PyException, false)
    serialize_py(s, x.v)
end

S.deserialize(s::S.AbstractSerializer, ::Type{PyException}) = PyException(deserialize_py(s))

### PyArray
#
# This type holds a pointer and a handle (usually a python memoryview or capsule) which are
# not serializable by default, and even if they were would not be consistent after
# serializing each field independently. So we just serialize the wrapped Python object.

function S.serialize(s::S.AbstractSerializer, x::PyArray)
    S.serialize_type(s, typeof(x), false)
    serialize_py(s, x.py)
end

function S.deserialize(s::S.AbstractSerializer, ::Type{T}) where {T<:PyArray}
    # TODO: set buffer and array args too?
    T(deserialize_py(s); copy = false)
end

end
