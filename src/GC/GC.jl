"""
    module PythonCall.GC

Garbage collection of Python objects.

See [`gc`](@ref).
"""
module GC

using ..C: C

const QUEUE = (; items = C.PyPtr[], lock = Threads.SpinLock())
const HOOK = WeakRef()

"""
    PythonCall.GC.disable()

Do nothing.

!!! note

    Historically this would disable the PythonCall garbage collector. This was required
    for safety in multi-threaded code but is no longer needed, so this is now a no-op.
"""
function disable()
    Base.depwarn(
        "disabling the PythonCall GC is no longer needed for thread-safety",
        :disable,
    )
    nothing
end

"""
    PythonCall.GC.enable()

Do nothing.

!!! note

    Historically this would enable the PythonCall garbage collector. This was required
    for safety in multi-threaded code but is no longer needed, so this is now a no-op.
"""
function enable()
    Base.depwarn(
        "disabling the PythonCall GC is no longer needed for thread-safety",
        :enable,
    )
    nothing
end

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
    Base.@lock QUEUE.lock begin
        for ptr in QUEUE.items
            if ptr != C.PyNULL
                C.Py_DecRef(ptr)
            end
        end
        empty!(QUEUE.items)
    end
    nothing
end

function enqueue(ptr::C.PyPtr)
    # If the ptr is NULL there is nothing to free.
    # If C.CTX.is_initialized is false then the Python interpreter hasn't started yet
    # or has been finalized; either way attempting to free will cause an error.
    if ptr != C.PyNULL && C.CTX.is_initialized
        if C.PyGILState_Check() == 1
            # If the current thread holds the GIL, then we can immediately free.
            C.Py_DecRef(ptr)
            # We may as well also free any other enqueued objects.
            if !isempty(QUEUE.items)
                unsafe_free_queue()
            end
        else
            # Otherwise we push the pointer onto the queue to be freed later, either:
            # (a) If a future Python object is finalized on the thread holding the GIL
            #     in the branch above.
            # (b) If the GCHook() object below is finalized in an ordinary GC.
            # (c) If the user calls PythonCall.GC.gc().
            Base.@lock QUEUE.lock push!(QUEUE.items, ptr)
        end
    end
    nothing
end

function enqueue_all(ptrs)
    if any(!=(C.PYNULL), ptrs) && C.CTX.is_initialized
        if C.PyGILState_Check() == 1
            for ptr in ptrs
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
            if !isempty(QUEUE.items)
                unsafe_free_queue()
            end
        else
            Base.@lock QUEUE.lock append!(QUEUE.items, ptrs)
        end
    end
    nothing
end

"""
    GCHook()

An immortal object which frees any pending Python objects when Julia's GC runs.

This works by creating it but not holding any strong reference to it, so it is eligible
to be finalized by Julia's GC. The finalizer empties the PythonCall GC queue if
possible. The finalizer also re-attaches itself, so the object does not actually get
collected and so the finalizer will run again at next GC.
"""
mutable struct GCHook
    function GCHook()
        finalizer(_gchook_finalizer, new())
    end
end

function _gchook_finalizer(x)
    if C.CTX.is_initialized
        finalizer(_gchook_finalizer, x)
        if !isempty(QUEUE.items) && C.PyGILState_Check() == 1
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
