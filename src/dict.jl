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

pydict(; kwargs...) = isempty(kwargs) ? pynew(errcheck(C.PyDict_New())) : pystrdict_fromiter(kwargs)
pydict(x) = ispy(x) ? pybuiltins.dict(x) : pydict_fromiter(x)
pydict(x::NamedTuple) = pydict(; x...)
export pydict
