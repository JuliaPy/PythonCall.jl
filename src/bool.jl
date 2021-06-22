pybool(x::Bool=false) = Py(x ? pybulitins.True : pybuiltins.False)
pybool(x::Number) = pybool(!iszero(x))
pybool(x) = pybulitins.bool(x)
export pybool
