using BenchmarkTools
using PythonCall
using PythonCall: pyimport, pydict, pystr, pyrange

# The same suite runs against main and v1, which use different names.
const benchmark_pydel = isdefined(PythonCall, :pydel!) ? PythonCall.pydel! : PythonCall.unsafe_pydel

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
        benchmark_pydel(k)
        benchmark_pydel(r)
        benchmark_pydel(v)
        benchmark_pydel(i)
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
                $(use_pydel ? :(@jl benchmark_pydel(i)) : :(nothing))
            end
            x
        end
    end
end

SUITE["basic"]["@py"]["pydict"]["init"] = @benchmarkable test_atpy(Val(false))
SUITE["basic"]["@py"]["pydict"]["pydel"] = @benchmarkable test_atpy(Val(true))


include("gcbench.jl")
using .GCBench: append_lots

SUITE["gc"]["full"] = @benchmarkable(
    GC.gc(true),
    setup=(GC.gc(true); append_lots(size=159)),
    seconds=30,
    evals=1,
)
