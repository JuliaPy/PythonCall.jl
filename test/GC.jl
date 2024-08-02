@testitem "201: GC segfaults" begin
    # https://github.com/JuliaPy/PythonCall.jl/issues/201
    # This should not segfault!
    cmd = Base.julia_cmd()
    path = joinpath(@__DIR__, "finalize_test_script.jl")
    p = run(`$cmd -t2 --project $path`)
    @test p.exitcode == 0
end

@testitem "GC.gc()" begin
    let
        pyobjs = map(pylist, 1:100)
        Threads.@threads for obj in pyobjs
            finalize(obj)
        end
    end
    Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
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
    Threads.nthreads() > 1 && @test !isempty(PythonCall.GC.QUEUE.items)
    GC.gc()
    @test isempty(PythonCall.GC.QUEUE.items)
end
