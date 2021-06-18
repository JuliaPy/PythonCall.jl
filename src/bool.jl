pybool(x::Bool=false) = Py(x ? pyTrue : pyFalse)
pybool(x::Number) = pybool(!iszero(x))
pybool(x) = pybooltype(x)
export pybool
