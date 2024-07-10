using PythonCall

# This would consistently segfault pre-GC-thread-safety
let
    pyobjs = map(pylist, 1:100)
    Threads.@threads for obj in pyobjs
        PythonCall.Core.py_finalizer(obj)
    end
end
