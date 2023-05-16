function fill_dict(n::Integer)
    x = pydict()
    
    for i in pyrange(n)
        x[pystr(i)] = i + py_random.random()
    end
    
    return x
end

function fill_dict_pydel(n::Integer)
    x = pydict()

    for i in pyrange(n)
        k = pystr(i)
        r = py_random.random()
        v = i + r

        x[k] = v

        pydel!(k)
        pydel!(r)
        pydel!(v)
        pydel!(i)
    end

    return x
end

SUITE["fill_dict"]       = @benchmarkable(fill_dict(1_000))
SUITE["fill_dict_pydel"] = @benchmarkable(fill_dict_pydel(1_000))
