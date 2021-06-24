function pytuple_setitem(xs::Py, i, x)
    x_ = Py(x)
    err = C.PyTuple_SetItem(getptr(xs), i, getptr(x_))
    pystolen!(x_)
    errcheck(err)
    return xs
end

function pytuple_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known, e.g. Tuple, Pair, Vector
        ans = pynulltuple(length(xs))
        for (i, x) in enumerate(xs)
            pytuple_setitem(ans, i-1, x)
        end
        return ans
    else
        # length unknown
        xs_ = pylist_fromiter(xs)
        ans = pylist_astuple(xs_)
        pydel!(xs_)
        return ans
    end
end

pynulltuple(len=0) = pynew(errcheck(C.PyTuple_New(len)))

pytuple() = pynulltuple(0)
pytuple(x) = ispy(x) ? pybuiltins.tuple(x) : pytuple_fromiter(x)
export pytuple
