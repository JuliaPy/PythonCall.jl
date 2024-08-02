@testitem "GC.gc()" begin
    let
        pyobjs = map(pylist, 1:100)
        Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    # The GC sometimes actually frees everything before this line.
    # We can uncomment this line if we GIL.@release the above block once we have it.
    # Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
    PythonCall.GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end

@testitem "GC.GCHook" begin
    let
        pyobjs = map(pylist, 1:100)
        Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    # The GC sometimes actually frees everything before this line.
    # We can uncomment this line if we GIL.@release the above block once we have it.
    # Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
    GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end
