function pylist_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known
        ans = setptr!(pynew(), errcheck(C.PyList_New(length(xs))))
        for (i, x) in enumerate(xs)
            t = Py(x)
            err = C.PyList_SetItem(getptr(ans), i-1, getptr(t))
            pystolen!(t)
            errcheck(err)
        end
        return ans
    else
        # length unknown
        ans = pylist()
        for x in xs
            errcheck(@autopy x C.PyList_Append(getptr(ans), getptr(x_)))
        end
        return ans
    end
end

pylist() = setptr!(pynew(), errcheck(C.PyList_New(0)))
pylist(x) = ispy(x) ? pylisttype(x) : pylist_fromiter(x)
export pylist

pylist_astuple(x) = setptr!(pynew(), errcheck(@autopy x C.PyList_AsTuple(getptr(x_))))
