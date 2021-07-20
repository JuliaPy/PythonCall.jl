# :PySet_New => (PyPtr,) => PyPtr,
# :PyFrozenSet_New => (PyPtr,) => PyPtr,
# :PySet_Add => (PyPtr, PyPtr) => Cint,

pyset_add(set::Py, x) = (errcheck(@autopy x C.PySet_Add(getptr(set), getptr(x_))); set)

function pyset_update_fromiter(set::Py, xs)
    for x in xs
        pyset_add(set, x)
    end
    return set
end
pyset_fromiter(xs) = pyset_update_fromiter(pyset(), xs)
pyfrozenset_fromiter(xs) = pyset_update_fromiter(pyfrozenset(), xs)

"""
    pyset(x=())

Convert `x` to a Python `set`.

If `x` is a Python object, this is equivalent to `set(x)` in Python.
Otherwise `x` must be iterable.
"""
pyset() = pynew(errcheck(C.PySet_New(C.PyNULL)))
pyset(x) = ispy(x) ? pybuiltins.set(x) : pyset_fromiter(x)
export pyset

"""
    pyfrozenset(x=())

Convert `x` to a Python `frozenset`.

If `x` is a Python object, this is equivalent to `frozenset(x)` in Python.
Otherwise `x` must be iterable.
"""
pyfrozenset() = pynew(errcheck(C.PyFrozenSet_New(C.PyNULL)))
pyfrozenset(x) = ispy(x) ? pybuiltins.frozenset(x) : pyfrozenset_fromiter(x)
export pyfrozenset
