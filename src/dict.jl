function pydict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        errcheck(@autopy k v C.PyDict_SetItem(getptr(ans), getptr(k_), getptr(v_)))
    end
    return ans
end

function pystrdict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        k2 = string(k)
        errcheck(@autopy k2 v C.PyDict_SetItem(getptr(ans), getptr(k2_), getptr(v_)))
    end
    return ans
end

pydict(; kwargs...) = isempty(kwargs) ? pynew(errcheck(C.PyDict_New())) : pystrdict_fromiter(kwargs)
pydict(x) = ispy(x) ? pybuiltins.dict(x) : pydict_fromiter(x)
pydict(x::NamedTuple) = pydict(; x...)
export pydict
