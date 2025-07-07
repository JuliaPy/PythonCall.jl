@testitem "unlock and lock" begin
    # This calls Python's time.sleep(1) twice concurrently. Since sleep() unlocks the
    # GIL, these can happen in parallel if Julia has at least 2 threads.
    function threaded_sleep()
        PythonCall.GIL.unlock() do
            Threads.@threads for i = 1:2
                PythonCall.GIL.lock() do
                    pyimport("time").sleep(1)
                end
            end
        end
    end
    # one run to ensure it's compiled
    threaded_sleep()
    # now time it
    t = @timed threaded_sleep()
    # if we have at least 2 threads, the sleeps run in parallel and take about a second
    if Threads.nthreads() ≥ 2
        @test 0.9 < t.time < 1.2
    end

    @test PythonCall.GIL.hasgil()
    PythonCall.GIL.unlock() do
        @test !Base.islocked(PythonCall.GIL._jl_gil_lock)
        PythonCall.GIL.lock() do
            @test Base.islocked(PythonCall.GIL._jl_gil_lock)
        end
    end
end

@testitem "@unlock and @lock" begin
    # This calls Python's time.sleep(1) twice concurrently. Since sleep() unlocks the
    # GIL, these can happen in parallel if Julia has at least 2 threads.
    function threaded_sleep()
        PythonCall.GIL.@unlock Threads.@threads for i = 1:2
            PythonCall.GIL.@lock pyimport("time").sleep(1)
        end
    end
    # one run to ensure it's compiled
    threaded_sleep()
    # now time it
    t = @timed threaded_sleep()
    # if we have at least 2 threads, the sleeps run in parallel and take about a second
    if Threads.nthreads() ≥ 2
        @test 0.9 < t.time < 1.2
    end

    @test PythonCall.GIL.hasgil()
    PythonCall.GIL.@unlock begin
        @test !Base.islocked(PythonCall.GIL._jl_gil_lock)
        PythonCall.GIL.@lock begin
            @test Base.islocked(PythonCall.GIL._jl_gil_lock)
        end
    end

end
