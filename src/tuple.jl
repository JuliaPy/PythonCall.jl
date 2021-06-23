function pytuple_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known
        ans = pynew(errcheck(C.PyTuple_New(length(xs))))
        for (i, x) in enumerate(xs)
            t = Py(x)
            err = C.PyTuple_SetItem(getptr(ans), i-1, getptr(t))
            pystolen!(t)
            errcheck(err)
        end
        return ans
    else
        # length unknown
        list = pylist_fromiter(xs)
        ans = pylist_astuple(list)
        pydel!(list)
        return ans
    end
end

pytuple() = pynew(errcheck(C.PyTuple_New(0)))
pytuple(x) = ispy(x) ? pybulitins.tuple(x) : pytuple_fromiter(x)
export pytuple
