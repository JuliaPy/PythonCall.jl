"""
    module PythonCall.GC

Garbage collection of Python objects.

See `disable` and `enable`.
"""
module GC

using ..C: C

# `ENABLED`: whether or not python GC is enabled, or paused to process later
const ENABLED = Threads.Atomic{Bool}(true)
# this event allows us to `wait` in a task until GC is re-enabled
# we have both this and `ENABLED` since there is no `isready(::Event)`
# for us to do a non-blocking check. Instead we must keep the event being triggered
# in-sync with `ENABLED[]`.
# We therefore modify both in `enable()` and `disable()` and nowhere else.
const ENABLED_EVENT = Threads.Event()

# this is the queue to process pointers for GC (`C.Py_DecRef`)
const QUEUE = Channel{C.PyPtr}(Inf)

# this is the task which performs GC from thread 1
const GC_TASK = Ref{Task}()

# This we use in testing to know when our GC is running
const GC_FINISHED = Threads.Condition()

# This is used for basic profiling
const SECONDS_SPENT_IN_GC = Threads.Atomic{Float64}()

"""
    PythonCall.GC.disable()

Disable the PythonCall garbage collector. This should generally not be required.

"""
function disable()
    ENABLED[] = false
    reset(ENABLED_EVENT)
    return
end

"""
    PythonCall.GC.enable()

Re-enable the PythonCall garbage collector. This should generally not be required.

"""
function enable()
    ENABLED[] = true
    notify(ENABLED_EVENT)
    return
end

function enqueue(ptr::C.PyPtr)
    if ptr != C.PyNULL && C.CTX.is_initialized    
        t = @elapsed put!(QUEUE, ptr)
        Threads.atomic_add!(SECONDS_SPENT_IN_GC, t)
    end
    return
end

function enqueue_all(ptrs)
    if C.CTX.is_initialized
        t = @elapsed for ptr in ptrs
            put!(QUEUE, ptr)
        end
        Threads.atomic_add!(SECONDS_SPENT_IN_GC, t)
    end
    return
end

# must only be called from thread 1 by the task in `GC_TASK[]`
function unsafe_process_queue!()
    if !isempty(QUEUE)
        C.with_gil(false) do
            while !isempty(QUEUE) && ENABLED[]
                # This should never block, since there should
                # only be one consumer
                # (we would like to not block while holding the GIL)
                ptr = take!(QUEUE)
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
        end
    end
    return nothing
end

function gc_loop()
    while true
        if ENABLED[] && !isempty(QUEUE)
            t = @elapsed unsafe_process_queue!()
            Threads.atomic_add!(SECONDS_SPENT_IN_GC, t)
            # just for testing purposes
            Base.@lock GC_FINISHED notify(GC_FINISHED)
        end
        # wait until there is both something to process
        # and GC is `enabled`
        wait(QUEUE)
        wait(ENABLED_EVENT)
    end
end

function launch_gc_task()
    if isassigned(GC_TASK) && Base.istaskstarted(GC_TASK[]) && !Base.istaskdone(GC_TASK[])
        throw(ConcurrencyViolationError("PythonCall GC task already running!"))
    end
    task = Task(gc_loop)
    task.sticky = VERSION >= v"1.7" # disallow task migration which was introduced in 1.7
    # ensure the task runs from thread 1
    ccall(:jl_set_task_tid, Cvoid, (Any, Cint), task, 0)
    schedule(task)
    if isdefined(Base, :errormonitor)
        Base.errormonitor(task)
    end
    GC_TASK[] = task
    task
end

function __init__()
    launch_gc_task()
    enable() # start enabled
    nothing
end

end # module GC
