@testitem "GC.gc()" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.Internals.GC.QUEUE.items)
    PythonCall.Internals.GC.gc()
    @test isempty(PythonCall.Internals.GC.QUEUE.items)
end

@testitem "GC.GCHook" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.Internals.GC.QUEUE.items)
    GC.gc()
    @test isempty(PythonCall.Internals.GC.QUEUE.items)
end
