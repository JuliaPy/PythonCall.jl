@testitem "region break and region" begin
    # This calls Python's time.sleep(1) twice concurrently. Since sleep() unlocks the
    # GIL, these can happen in parallel if Julia has at least 2 threads.
    function threaded_sleep()
        @pyregionbreak begin
            Threads.@threads :static for i = 1:2
                @pyregion begin
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
end

@testitem "@pyregionbreak and @pyregion" begin
    # This calls Python's time.sleep(1) twice concurrently. Since sleep() unlocks the
    # GIL, these can happen in parallel if Julia has at least 2 threads.
    function threaded_sleep()
        @pyregionbreak Threads.@threads :static for i = 1:2
            @pyregion pyimport("time").sleep(1)
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
end
