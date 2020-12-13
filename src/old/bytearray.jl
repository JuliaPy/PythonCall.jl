const pybytearraytype = pylazyobject(() -> pybuiltins.bytearray)
export pybytearraytype

pyisbytearray(o::PyObject) = pytypecheck(o, pybytearraytype)
export pyisbytearray

pybytearray(args...; opts...) = pybytearraytype(args...; opts...)
export pybytearray
