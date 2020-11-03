const pysettype = PyLazyObject(() -> pybuiltins.set)
const pyfrozensettype = PyLazyObject(() -> pybuiltins.frozenset)
export pysettype, pyfrozensettype

pyisset(o::AbstractPyObject) = pytypecheck(o, pysettype)
pyisfrozenset(o::AbstractPyObject) = pytypecheck(o, pyfrozensettype)
pyisanyset(o::AbstractPyObject) = pyisset(o) || pyisfrozenset(o)
export pyisset, pyisfrozenset, pyisanyset

pyset(args...; opts...) = pysettype(args...; opts...)
pyfrozenset(args...; opts...) = pyfrozensettype(args...; opts...)
export pyset, pyfrozenset
