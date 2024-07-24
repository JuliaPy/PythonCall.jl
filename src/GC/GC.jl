"""
    module PythonCall.GC

Garbage collection of Python objects.

See `disable` and `enable`.
"""
module GC

using ..C: C

const ENABLED = Ref(true)
const QUEUE = C.PyPtr[]

"""
    PythonCall.GC.disable()

Disable the PythonCall garbage collector.

This means that whenever a Python object owned by Julia is finalized, it is not immediately
freed but is instead added to a queue of objects to free later when `enable()` is called.

Like most PythonCall functions, you must only call this from the main thread.
"""
function disable()
    ENABLED[] = false
    return
end

"""
    PythonCall.GC.enable()

Re-enable the PythonCall garbage collector.

This frees any Python objects which were finalized while the GC was disabled, and allows
objects finalized in the future to be freed immediately.

Like most PythonCall functions, you must only call this from the main thread.
"""
function enable()
    ENABLED[] = true
    if !isempty(QUEUE) && C.PyGILState_Check() == 1
        free_queue()
    end
    return
end

function free_queue()
    for ptr in QUEUE
        if ptr != C.PyNULL
            C.Py_DecRef(ptr)
        end
    end
    empty!(QUEUE)
    nothing
end

function gc()
    if ENABLED[] && C.PyGILState_Check() == 1
        free_queue()
        true
    else
        false
    end
end

function enqueue(ptr::C.PyPtr)
    if ptr != C.PyNULL && C.CTX.is_initialized
        if ENABLED[] && C.PyGILState_Check() == 1
            C.Py_DecRef(ptr)
            isempty(QUEUE) || free_queue()
        else
            push!(QUEUE, ptr)
        end
    end
    return
end

function enqueue_all(ptrs)
    if C.CTX.is_initialized
        if ENABLED[] && C.PyGILState_Check() == 1
            for ptr in ptrs
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
            isempty(QUEUE) || free_queue()
        else
            append!(QUEUE, ptrs)
        end
    end
    return
end

mutable struct GCHook
    function GCHook()
        finalizer(_gchook_finalizer, new())
    end
end

function _gchook_finalizer(x)
    gc()
    finalizer(_gchook_finalizer, x)
    nothing
end

function __init__()
    GCHook()
    nothing
end

end # module GC
