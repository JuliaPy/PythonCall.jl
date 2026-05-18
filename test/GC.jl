@testitem "GC.gc()" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock begin
            Threads.@threads for obj in pyobjs
                finalize(obj)
            end
            Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.GC.QUEUE.items)
    PythonCall.GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end

@testitem "GC.GCHook" begin
    let
        pyobjs = map(pylist, 1:100)
        PythonCall.GIL.@unlock begin
            Threads.@threads for obj in pyobjs
                finalize(obj)
            end
            Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
        end
    end
    Threads.nthreads() > 1 &&
        VERSION >= v"1.10.0-" &&
        @test !isempty(PythonCall.GC.QUEUE.items)
    GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end
