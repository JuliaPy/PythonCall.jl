const pysettype = pylazyobject(() -> pybuiltins.set)
const pyfrozensettype = pylazyobject(() -> pybuiltins.frozenset)
export pysettype, pyfrozensettype

pyisset(o::PyObject) = pytypecheck(o, pysettype)
pyisfrozenset(o::PyObject) = pytypecheck(o, pyfrozensettype)
pyisanyset(o::PyObject) = pyisset(o) || pyisfrozenset(o)
export pyisset, pyisfrozenset, pyisanyset

pyset(args...; opts...) = pysettype(args...; opts...)
pyset(x::Union{Tuple,AbstractVector}) = pyset(pylist(x))
pyfrozenset(args...; opts...) = pyfrozensettype(args...; opts...)
pyfrozenset(x::Union{Tuple,AbstractVector}) = pyfrozenset(pylist(x))
export pyset, pyfrozenset
