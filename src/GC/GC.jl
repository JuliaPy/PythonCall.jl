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

const LOGGING_ENABLED = Ref{Bool}(false)

"""
    PythonCall.GC.enable_logging(enable=true)

Enables printed logging (similar to Julia's `GC.enable_logging`).
"""
function enable_logging(enable=true)
    LOGGING_ENABLED[] = enable
    return nothing
end

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

function enqueue_wrapper(f, g)
    t = @elapsed begin
        if C.CTX.is_initialized
            # Eager path: if we are already on thread 1,
            # we eagerly decrement
            handled = false
            if ENABLED[] && Threads.threadid() == 1
                # temporarily disable thread migration to be sure
                # we call `C.Py_DecRef` from thread 1
                old_sticky = current_task().sticky
                if !old_sticky
                    current_task().sticky = true
                end
                if Threads.threadid() == 1
                    f()
                    # if ptr != C.PyNULL
                    # C.Py_DecRef(ptr)
                    # end
                    handled = true
                end
                if !old_sticky
                    current_task().sticky = old_sticky
                end
            end
            if !handled
                g()
                # if ptr != C.PyNULL
                # put!(QUEUE, ptr)
                # end
            end
        end
    end
    Threads.atomic_add!(SECONDS_SPENT_IN_GC, t)
    return
end

function enqueue(ptr::C.PyPtr)
    # if we are on thread 1:
    f = () -> begin
        C.with_gil(false) do
            if ptr != C.PyNULL
                C.Py_DecRef(ptr)
            end
        end
    end
    # otherwise:
    g = () -> begin
        if ptr != C.PyNULL
            put!(QUEUE, ptr)
        end
    end
    enqueue_wrapper(f, g)
end

function enqueue_all(ptrs)
    # if we are on thread 1:
    f = () -> begin
        C.with_gil(false) do
            for ptr in ptrs
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                end
            end
        end
    end
    # otherwise:
    g = () -> begin
        for ptr in ptrs
            if ptr != C.PyNULL
                put!(QUEUE, ptr)
            end
        end
    end
    enqueue_wrapper(f, g)
end

# function enqueue_all(ptrs)
#     t = @elapsed begin
#         if C.CTX.is_initialized
#             # Eager path: if we are already on thread 1,
#             # we eagerly decrement
#             handled = false
#             if ENABLED[] && Threads.threadid() == 1
#                 # temporarily disable thread migration to be sure
#                 # we call `C.Py_DecRef` from thread 1
#                 old_sticky = current_task().sticky
#                 if !old_sticky
#                     current_task().sticky = true
#                 end
#                 if Threads.threadid() == 1
#                     for ptr in ptrs
#                         if ptr != C.PyNULL
#                             C.Py_DecRef(ptr)
#                         end
#                     end
#                     handled = true
#                 end
#                 if !old_sticky
#                     current_task().sticky = old_sticky
#                 end
#             end
#             if !handled
#                 for ptr in ptrs
#                     if ptr != C.PyNULL
#                         put!(QUEUE, ptr)
#                     end
#                 end
#             end
#         end
#     end
#     Threads.atomic_add!(SECONDS_SPENT_IN_GC, t)
#     return
# end

# must only be called from thread 1 by the task in `GC_TASK[]`
function unsafe_process_queue!()
    n = 0
    if !isempty(QUEUE)
        t = @elapsed C.with_gil(false) do
            while !isempty(QUEUE) && ENABLED[]
                # This should never block, since there should
                # only be one consumer
                # (we would like to not block while holding the GIL)
                ptr = take!(QUEUE)
                if ptr != C.PyNULL
                    C.Py_DecRef(ptr)
                    n += 1
                end
            end
        end
        if LOGGING_ENABLED[]
            Base.time_print(stdout, t; msg="Python GC ($n items)")
            println(stdout)
        end
    else
        t = 0.0
    end
    return t
end

function gc_loop()
    while true
        if ENABLED[] && !isempty(QUEUE)
            t = unsafe_process_queue!()
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
