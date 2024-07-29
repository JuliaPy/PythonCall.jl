using PythonCall

# This would consistently segfault pre-GC-thread-safety
let
    pyobjs = map(pylist, 1:100)
    Threads.@threads for obj in pyobjs
        finalize(obj)
    end
end

@show isready(PythonCall.GC.QUEUE)
GC.gc()
# with GCHook, the queue should be empty now (the above gc() triggered GCHook to clear the PythonCall QUEUE)
# without GCHook, gc() has no effect on the QUEUE
@show isready(PythonCall.GC.QUEUE)
GC.gc()
@show isready(PythonCall.GC.QUEUE)
GC.gc()
@show isready(PythonCall.GC.QUEUE)
# with GCHook this is not necessary, GC.gc() is enough
# without GCHook, this is required to free any objects in the PythonCall QUEUE
PythonCall.GC.gc()
@show isready(PythonCall.GC.QUEUE)
