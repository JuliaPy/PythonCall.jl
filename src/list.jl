pynulllist(len) = pynew(errcheck(C.PyList_New(len)))

function pylist_setitem(xs::Py, i, x)
    x_ = Py(x)
    err = C.PyList_SetItem(getptr(xs), i, getptr(x_))
    pystolen!(x_)
    errcheck(err)
    return xs
end

pylist_append(xs::Py, x) = errcheck(@autopy x C.PyList_Append(getptr(ans), getptr(x_)))

pylist_astuple(x) = pynew(errcheck(@autopy x C.PyList_AsTuple(getptr(x_))))

function pylist_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known
        ans = pynulllist(length(xs))
        for (i, x) in enumerate(xs)
            pylist_setitem(ans, i-1, x)
        end
        return ans
    else
        # length unknown
        ans = pynulllist(0)
        for x in xs
            pylist_append(ans, x)
        end
        return ans
    end
end

pylist() = pynulllist(0)
pylist(x) = ispy(x) ? pybuiltins.list(x) : pylist_fromiter(x)
export pylist
