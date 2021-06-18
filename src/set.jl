# :PySet_New => (PyPtr,) => PyPtr,
# :PyFrozenSet_New => (PyPtr,) => PyPtr,
# :PySet_Add => (PyPtr, PyPtr) => Cint,

function pyset_update_fromiter!(set::Py, xs)
    for x in xs
        errcheck(@autopy x C.PySet_Add(getptr(set), getptr(x_)))
    end
    return set
end
pyset_fromiter(xs) = pyset_update_fromiter!(pyset(), xs)
pyfrozenset_fromiter(xs) = pyset_update_fromiter!(pyfrozenset(), xs)

pyset() = setptr!(pynew(), errcheck(C.PySet_New(C.PyNULL)))
pyset(x) = ispy(x) ? pysettype(x) : pyset_fromiter(x)
export pyset

pyfrozenset() = setptr!(pynew(), errcheck(C.PyFrozenSet_New(C.PyNULL)))
pyfrozenset(x) = ispy(x) ? pyfrozensettype(x) : pyfrozenset_fromiter(x)
export pyfrozenset
