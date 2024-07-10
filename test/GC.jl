@testset "201: GC segfaults" begin
    # https://github.com/JuliaPy/PythonCall.jl/issues/201
    # This should not segfault!
    cmd = Base.julia_cmd()
    path = joinpath(@__DIR__, "finalize_test_script.jl")
    p = run(`$cmd -t2 --project $path`)
    @test p.exitcode == 0
end

@testset "GC.enable() and GC.disable()" begin
    PythonCall.GC.disable()
    try
        # Put some stuff in the GC queue
        let
            pyobjs = map(pylist, 1:100)
            foreach(PythonCall.Core.py_finalizer, pyobjs)
        end
        # Since GC is disabled, we should have built up entries in the queue
        # (there is no race since we push to the queue eagerly)
        @test !isempty(PythonCall.GC.QUEUE)
        # now, setup a task so we can know once GC has triggered.
        # We start waiting *before* enabling GC, so we know we'll catch
        # the notification that GC finishes
        # We also don't enable GC until we get confirmation via `is_waiting[]`.
        # At this point we have acquired the lock for `PythonCall.GC.GC_FINISHED`.
        # We won't relinquish it until the next line where we `wait`,
        # at which point the GC can trigger it. Therefore we should be certain
        # that the ordering is correct and we can't miss the event.
        is_waiting = Threads.Atomic{Bool}(false)
        rdy = Threads.@spawn begin
            Base.@lock PythonCall.GC.GC_FINISHED begin
                is_waiting[] = true
                wait(PythonCall.GC.GC_FINISHED)
            end
        end
        # Wait until the task starts
        ret = timedwait(5) do
            istaskstarted(rdy) && is_waiting[]
        end
        @test ret == :ok
        # Now, re-enable GC
        PythonCall.GC.enable()
        # Wait for GC to run
        wait(rdy)
        # There should be nothing left in the queue, since we fully process it.
        @test isempty(PythonCall.GC.QUEUE)
    finally
        # Make sure we re-enable GC regardless of whether our tests pass
        PythonCall.GC.enable()
    end
end

@test_throws ConcurrencyViolationError("PythonCall GC task already running!") PythonCall.GC.launch_gc_task()
