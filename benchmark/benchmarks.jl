using BenchmarkTools
using PythonCall
using PythonCall: unsafe_pydel, pyimport, pydict, pystr, pyrange

const SUITE = BenchmarkGroup()

function test_pydict_init()
    random = pyimport("random").random
    x = pydict()
    for i in pyrange(1000)
        x[pystr(i)] = i + random()
    end
    return x
end

SUITE["basic"]["julia"]["pydict"]["init"] = @benchmarkable test_pydict_init()

function test_pydict_pydel()
    random = pyimport("random").random
    x = pydict()
    for i in pyrange(1000)
        k = pystr(i)
        r = random()
        v = i + r
        x[k] = v
        unsafe_pydel(k)
        unsafe_pydel(r)
        unsafe_pydel(v)
        unsafe_pydel(i)
    end
    return x
end

SUITE["basic"]["julia"]["pydict"]["pydel"] = @benchmarkable test_pydict_pydel()

@generated function test_atpy(::Val{use_pydel}) where {use_pydel}
    quote
        @py begin
            import random: random
            x = {}
            for i in range(1000)
                x[str(i)] = i + random()
                $(use_pydel ? :(@jl PythonCall.unsafe_pydel(i)) : :(nothing))
            end
            x
        end
    end
end

SUITE["basic"]["@py"]["pydict"]["init"] = @benchmarkable test_atpy(Val(false))
SUITE["basic"]["@py"]["pydict"]["pydel"] = @benchmarkable test_atpy(Val(true))

function test_region_pythoncall(x, n)
    ans = 0
    for _ = 1:n
        ans += pylen(x)
    end
    return ans
end

function test_region_pythoncall_outer(x, n)
    @pyregion begin
        ans = 0
        for _ = 1:n
            ans += pylen(x)
        end
        return ans
    end
end

function test_region_capi(x, n)
    @pyregion begin
        ans = 0
        for _ = 1:n
            ans += PythonCall.C.PyObject_Length(x)
        end
        return ans
    end
end

const REGION_BENCHMARK_LENGTH = 1000

SUITE["region"]["pythoncall"] = @benchmarkable(
    test_region_pythoncall(x, REGION_BENCHMARK_LENGTH),
    setup=(x = pytuple((1, 2, 3))),
)
SUITE["region"]["pythoncall_outer"] = @benchmarkable(
    test_region_pythoncall_outer(x, REGION_BENCHMARK_LENGTH),
    setup=(x = pytuple((1, 2, 3))),
)
SUITE["region"]["capi_outer"] = @benchmarkable(
    test_region_capi(x, REGION_BENCHMARK_LENGTH),
    setup=(x = pytuple((1, 2, 3))),
)


include("gcbench.jl")
using .GCBench: append_lots

SUITE["gc"]["full"] = @benchmarkable(
    GC.gc(true),
    setup=(GC.gc(true); append_lots(size=159)),
    seconds=30,
    evals=1,
)
