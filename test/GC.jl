@testitem "GC.gc()" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    PythonCall.GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end

@testitem "GC.GCHook" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end
