using PythonCall, Test

function wait_for_queue_to_be_empty()
    ret = timedwait(5) do
        isempty(PythonCall.GC.QUEUE)
    end
    ret === :ok || error("QUEUE still not empty")
end

# https://github.com/JuliaCI/GCBenchmarks/blob/fc288c696381ebfdef8f002168addd0ec1b08e34/benches/serial/append/append.jl
macro gctime(ex)
    quote
        local prior = PythonCall.GC.SECONDS_SPENT_IN_GC[]
        local ret = @timed $(esc(ex))
        Base.time_print(stdout, ret.time * 1e9, ret.gcstats.allocd, ret.gcstats.total_time, Base.gc_alloc_count(ret.gcstats); msg="Runtime")
        println(stdout)
        local waiting = @elapsed wait_for_queue_to_be_empty()
        local after = PythonCall.GC.SECONDS_SPENT_IN_GC[]
        Base.time_print(stdout, (after - prior) * 1e9; msg="Python GC time")
        println(stdout)
        Base.time_print(stdout, waiting * 1e9; msg="Python GC time (waiting)")
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

GC.enable_logging(false)
PythonCall.GC.enable_logging(false)
@time "Total" begin
    @gctime append_lots()
    @time "Next full GC" begin
        GC.gc(true)
        wait_for_queue_to_be_empty()
    end
end
