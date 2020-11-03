const pybytearraytype = PyLazyObject(() -> pybuiltins.bytearray)
export pybytearraytype

pyisbytearray(o::AbstractPyObject) = pytypecheck(o, pybytearraytype)
export pyisbytearray

pybytearray(args...; opts...) = pybytearraytype(args...; opts...)
export pybytearray
