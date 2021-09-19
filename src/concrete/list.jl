pynulllist(len) = pynew(errcheck(C.PyList_New(len)))

function pylist_setitem(xs::Py, i, x)
    x_ = Py(x)
    err = C.PyList_SetItem(getptr(xs), i, getptr(x_))
    pystolen!(x_)
    errcheck(err)
    return xs
end

pylist_append(xs::Py, x) = errcheck(@autopy x C.PyList_Append(getptr(xs), getptr(x_)))

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

"""
    pylist(x=())

Convert `x` to a Python `list`.

If `x` is a Python object, this is equivalent to `list(x)` in Python.
Otherwise `x` must be iterable.
"""
pylist() = pynulllist(0)
pylist(x) = ispy(x) ? pybuiltins.list(x) : pylist_fromiter(x)
export pylist

"""
    pycollist(x::AbstractArray)

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of columns.
"""
function pycollist(x::AbstractArray{T,N}) where {T,N}
    ndims(x) == 0 && return Py(x[])
    d = N
    ax = axes(x, d)
    ans = pynulllist(length(ax))
    for (i, j) in enumerate(ax)
        y = pycollist(selectdim(x, d, j))
        pylist_setitem(ans, i-1, y)
        pydel!(y)
    end
    return ans
end
export pycollist

"""
    pyrowlist(x::AbstractArray)

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of rows.
"""
function pyrowlist(x::AbstractArray{T,N}) where {T,N}
    ndims(x) == 0 && return Py(x[])
    d = 1
    ax = axes(x, d)
    ans = pynulllist(length(ax))
    for (i, j) in enumerate(ax)
        y = pyrowlist(selectdim(x, d, j))
        pylist_setitem(ans, i-1, y)
        pydel!(y)
    end
    return ans
end
export pyrowlist
