using PythonCall

# This would consistently segfault pre-GC-thread-safety
function test()
    pyobjs = map(pyint, 1:10)
    Threads.@threads for i = 1:10
        finalize(pyobjs[i])
    end
    # The following loop is a workaround and can be removed if the issue is fixed:
    # https://github.com/JuliaLang/julia/issues/40626#issuecomment-1054890774
    Threads.@threads :static for _ = 1:Threads.nthreads()
        Timer(Returns(nothing), 0; interval = 1)
    end
    nothing
end

function decrefs()
    n = PythonCall.Core.NUM_DECREFS[]
    PythonCall.Core.NUM_DECREFS[] = 0
    n
end

GC.gc()
decrefs()
println("test()")
test()
println("  decrefs: ", decrefs())
println("gc(false)")
GC.gc(false)
println("  decrefs: ", decrefs())
println("gc(false)")
GC.gc(false)
println("  decrefs: ", decrefs())
println("gc()")
GC.gc()
println("  decrefs: ", decrefs())
println("gc()")
GC.gc()
println("  decrefs: ", decrefs())
println("gc()")
GC.gc()
println("  decrefs: ", decrefs())
println("gc()")
GC.gc()
println("  decrefs: ", decrefs())
println("gc()")
GC.gc()
println("  decrefs: ", decrefs())
