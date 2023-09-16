pydict_setitem(x::Py, k, v) = errcheck(@autopy k v C.PyDict_SetItem(getptr(x), getptr(k_), getptr(v_)))

function pydict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        pydict_setitem(ans, k, v)
    end
    return ans
end

function pystrdict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        pydict_setitem(ans, string(k), v)
    end
    return ans
end

"""
    pydict(x)
    pydict(; x...)

Convert `x` to a Python `dict`. In the second form, the keys are strings.

If `x` is a Python object, this is equivalent to `dict(x)` in Python.
Otherwise `x` must iterate over key-value pairs.
"""
pydict(; kwargs...) = isempty(kwargs) ? pynew(errcheck(C.PyDict_New())) : pystrdict_fromiter(kwargs)
pydict(x) = ispy(x) ? pybuiltins.dict(x) : pydict_fromiter(x)
pydict(x::NamedTuple) = pydict(; x...)
export pydict
