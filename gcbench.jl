using PythonCall, Test

# https://github.com/JuliaCI/GCBenchmarks/blob/fc288c696381ebfdef8f002168addd0ec1b08e34/benches/serial/append/append.jl
macro gctime(ex)
    quote
        local prior = PythonCall.GC.SECONDS_SPENT_IN_GC[]
        local ret = @timed $(esc(ex))
        Base.time_print(stdout, ret.time * 1e9, ret.gcstats.allocd, ret.gcstats.total_time, Base.gc_alloc_count(ret.gcstats); msg="Runtime")
        local after = PythonCall.GC.SECONDS_SPENT_IN_GC[]
        println(stdout)
        Base.time_print(stdout, (after - prior) * 1e9; msg="Python GC time")
        println(stdout)
        ret.value
    end
end

function append_lots(iters=100 * 1024, size=1596)
    v = pylist()
    for i = 1:iters
        v.append(pylist(rand(size)))
    end
    return v
end

@time "Total" begin
    @gctime append_lots()
    @time "Next full GC" begin
        GC.gc(true)
    end
end
