"""
    module PythonCall.GC

Garbage collection of Python objects.

See [`enable`](@ref), [`disable`](@ref) and [`gc`](@ref).
"""
module GC

using ..C: C

const QUEUE = Channel{C.PyPtr}(Inf)
const HOOK = WeakRef()

"""
    PythonCall.GC.disable()

Do nothing.

!!! note

    Historically this would disable the PythonCall garbage collector. This was required
    for safety in multi-threaded code but is no longer needed, so this is now a no-op.
"""
disable() = nothing

"""
    PythonCall.GC.enable()

Do nothing.

!!! note

    Historically this would enable the PythonCall garbage collector. This was required
    for safety in multi-threaded code but is no longer needed, so this is now a no-op.
"""
enable() = nothing

"""
    PythonCall.GC.gc()

Free any Python objects waiting to be freed.

These are objects that were finalized from a thread that was not holding the Python
GIL at the time.

Like most PythonCall functions, this must only be called from the main thread (i.e. the
thread currently holding the Python GIL.)
"""
function gc()
    if C.CTX.is_initialized
        unsafe_free_queue()
    end
    nothing
end

function unsafe_free_queue()
    if isready(QUEUE)
        @lock QUEUE while isready(QUEUE)
            ptr = take!(QUEUE)
            if ptr != C.PyNULL
                C.Py_DecRef(ptr)
            end
        end
    end
    nothing
end

function enqueue(ptr::C.PyPtr)
    if ptr != C.PyNULL && C.CTX.is_initialized
        if C.PyGILState_Check() == 1
            C.Py_DecRef(ptr)
            unsafe_free_queue()
        else
            put!(QUEUE, ptr)
        end
    end
    nothing
end

function enqueue_all(ptrs)
    if any(ptr -> ptr != C.PYNULL, ptrs) && C.CTX.is_initialized
        if C.PyGILState_Check() == 1
            for ptr in ptrs
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
            unsafe_free_queue()
        else
            for ptr in ptrs
                put!(QUEUE, ptr)
            end
        end
    end
    nothing
end

mutable struct GCHook
    function GCHook()
        finalizer(_gchook_finalizer, new())
    end
end

function _gchook_finalizer(x)
    if C.CTX.is_initialized
        finalizer(_gchook_finalizer, x)
        if isready(QUEUE) && C.PyGILState_Check() == 1
            unsafe_free_queue()
        end
    end
    nothing
end

function __init__()
    HOOK.value = GCHook()
    nothing
end

end # module GC
