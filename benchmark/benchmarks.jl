using PythonCall, BenchmarkTools

const SUITE = BenchmarkGroup()

function test_pydict_init()
    random = pyimport("random").random
    x = pydict()
    for i in pyrange(1000)
        x[pystr(i)] = i + random()
    end
    return x
end

SUITE["pydict"]["init"] = @benchmarkable test_pydict_init()

function test_pydict_pydel()
    random = pyimport("random").random
    x = pydict()
    for i in pyrange(1000)
        k = pystr(i)
        r = random()
        v = i + r
        x[k] = v
        pydel!(k)
        pydel!(r)
        pydel!(v)
        pydel!(i)
    end
    return x
end

SUITE["pydict"]["pydel"] = @benchmarkable test_pydict_pydel()

@generated function test_atpy(::Val{use_pydel}) where {use_pydel}
    quote
        @py begin
            import random: random
            x = {}
            for i in range(1000)
                x[str(i)] = i + random()
                $(use_pydel ? :(@jl PythonCall.pydel!(i)) : :(nothing))
            end
            x
        end
    end
end

SUITE["@py"]["basic"] = @benchmarkable test_atpy(Val(false))
SUITE["@py"]["pydel"] = @benchmarkable test_atpy(Val(true))
