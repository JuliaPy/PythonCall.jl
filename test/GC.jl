using TestItemRunner

@testitem "GC.gc()" begin
    using PythonCall
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.GC.QUEUE.items)
    PythonCall.GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end

@testitem "GC.GCHook" begin
    using PythonCall
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.GC.QUEUE.items)
    GC.gc()

    # Unlock and relocking the ReentrantLock allows this test to pass
    # if _jl_gil_lock is locked on init
    # Base.unlock(PythonCall.GIL._jl_gil_lock)
    # Base.lock(PythonCall.GIL._jl_gil_lock)

    @test isempty(PythonCall.GC.QUEUE.items)
end
