"""Internal task-safe management of CPython thread states."""
module Region

using ..C: C
import ..PythonCall: @pyregion, @pyregionbreak

const PyThreadStatePtr = Ptr{Cvoid}

mutable struct ThreadState
    sem::Base.Semaphore
    tstate::PyThreadStatePtr
end
ThreadState() = ThreadState(Base.Semaphore(1), C_NULL)

mutable struct TaskState
    tstate::PyThreadStatePtr
    sem::Union{Nothing,Base.Semaphore}
    attached::Bool
    tid::Int
    oldsticky::Bool
end
TaskState() = TaskState(C_NULL, nothing, false, 0, false)

const THREAD_STATE = Base.OncePerThread(ThreadState)
const TASK_STATE = Base.OncePerTask(TaskState)
const INTERP = Ref{Ptr{Cvoid}}(C_NULL)

current_tstate() = C.PyThreadState_GetUnchecked()
has_tstate() = C.CTX.is_initialized && current_tstate() != C_NULL

function reset!(s::TaskState)
    s.tstate = C_NULL
    s.sem = nothing
    s.attached = false
    s.tid = 0
    return
end

function start_session!(s::TaskState)
    task = current_task()
    s.oldsticky = task.sticky
    task.sticky = true
    s.tid = Threads.threadid()
    ts = THREAD_STATE()
    s.sem = ts.sem
    return ts
end

check_thread(s::TaskState) = (@assert current_task().sticky && Threads.threadid() == s.tid)

function enter_region()
    s = TASK_STATE()
    if s.tstate != C_NULL
        check_thread(s)
        if s.attached
            return :noop
        end
        Base.acquire(s.sem::Base.Semaphore)
        try
            C.PyEval_RestoreThread(s.tstate)
            s.attached = true
        catch
            Base.release(s.sem::Base.Semaphore)
            rethrow()
        end
        return :detach
    end

    task = current_task()
    ts = try
        start_session!(s)
    catch
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
    acquired = false
    try
        Base.acquire(ts.sem)
        acquired = true
        current = current_tstate()
        if current != C_NULL
            s.tstate = current
            s.attached = true
            return :root_borrowed
        end
        if ts.tstate == C_NULL
            ts.tstate = C.PyThreadState_New(INTERP[])
            ts.tstate == C_NULL && error("PyThreadState_New failed")
        end
        s.tstate = ts.tstate
        C.PyEval_RestoreThread(s.tstate)
        s.attached = true
        return :root_owned
    catch
        acquired && Base.release(ts.sem)
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
end

function exit_region(token)
    token === :noop && return
    s = TASK_STATE()
    check_thread(s)
    if token === :detach
        saved = C.PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        Base.release(s.sem::Base.Semaphore)
    elseif token === :root_owned
        saved = C.PyEval_SaveThread()
        @assert saved == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        current_task().sticky = oldsticky
    elseif token === :root_borrowed
        @assert current_tstate() == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        current_task().sticky = oldsticky
    else
        error("invalid Python region token")
    end
    return
end

function enter_break()
    s = TASK_STATE()
    if s.tstate != C_NULL
        check_thread(s)
        !s.attached && return :noop
        saved = C.PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        Base.release(s.sem::Base.Semaphore)
        return :restore
    end
    current_tstate() == C_NULL && return :noop

    task = current_task()
    ts = try
        start_session!(s)
    catch
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
    acquired = false
    try
        Base.acquire(ts.sem)
        acquired = true
        current = current_tstate()
        if current == C_NULL
            Base.release(ts.sem)
            task.sticky = s.oldsticky
            reset!(s)
            return :noop
        end
        s.tstate = current
        saved = C.PyEval_SaveThread()
        @assert saved == current
        s.attached = false
        Base.release(ts.sem)
        return :root_restore
    catch
        acquired && Base.release(ts.sem)
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
end

function exit_break(token)
    token === :noop && return
    s = TASK_STATE()
    check_thread(s)
    Base.acquire(s.sem::Base.Semaphore)
    C.PyEval_RestoreThread(s.tstate)
    s.attached = true
    if token === :root_restore
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        current_task().sticky = oldsticky
    elseif token !== :restore
        error("invalid Python region-break token")
    end
    return
end

"""
    @pyregion expr

Mark `expr` as relatively straight-line, Python-heavy work. PythonCall APIs manage the
Python resources they need automatically; this optional region can amortize that work across
several operations. Regions nest freely. Put Julia code which deliberately yields, waits, or
blocks cooperatively in [`@pyregionbreak`](@ref).
"""
macro pyregion(ex)
    quote
        local token = $enter_region()
        try
            $(esc(ex))
        finally
            $exit_region(token)
        end
    end
end

"""
    @pyregionbreak expr

Mark `expr` as a section where Python interaction is absent or infrequent, allowing an
enclosing [`@pyregion`](@ref) to relinquish Python-related resources temporarily. Nested
PythonCall operations still work automatically, and both kinds of region nest freely.
"""
macro pyregionbreak(ex)
    quote
        local token = $enter_break()
        try
            $(esc(ex))
        finally
            $exit_break(token)
        end
    end
end

function __init__()
    current = current_tstate()
    current == C_NULL && error("Python initialization did not leave an attached thread state")
    INTERP[] = C.PyThreadState_GetInterpreter(current)
    INTERP[] == C_NULL && error("could not determine the Python interpreter")
    return
end

function start_runtime()
    if !C.CTX.is_embedded && !C.CTX.is_preinitialized
        C.PyEval_SaveThread()
        C.FINALIZE_HOOK[] = function ()
            ts = THREAD_STATE()
            ts.tstate == C_NULL && (ts.tstate = C.PyThreadState_New(INTERP[]))
            C.PyEval_RestoreThread(ts.tstate)
            C.Py_FinalizeEx() == -1 && @warn "Py_FinalizeEx() error"
        end
    end
    return
end

end
