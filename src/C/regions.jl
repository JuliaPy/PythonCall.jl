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

const THREAD_STATE = Utils.OncePerThread{ThreadState}(ThreadState)
const TASK_STATE = Utils.OncePerTask{TaskState}(TaskState)

current_tstate() = PyThreadState_GetUnchecked()
has_tstate() = CTX.is_initialized && current_tstate() != C_NULL

function reset!(s::TaskState)
    s.tstate = C_NULL
    s.sem = nothing
    s.attached = false
    s.tid = 0
    return
end

function start_session!(task::Task, s::TaskState)
    s.oldsticky = task.sticky
    task.sticky = true
    s.tid = Threads.threadid()
    ts = THREAD_STATE()
    s.sem = ts.sem
    return ts
end

function check_thread(task::Task, s::TaskState)
    @assert task.sticky
    @assert Threads.threadid() == s.tid
end

function enter_region(task::Task, s::TaskState)
    if s.tstate != C_NULL
        check_thread(task, s)
        if s.attached
            return :noop
        end
        Base.acquire(s.sem::Base.Semaphore)
        try
            PyEval_RestoreThread(s.tstate)
            s.attached = true
        catch
            Base.release(s.sem::Base.Semaphore)
            rethrow()
        end
        return :detach
    end

    ts = try
        start_session!(task, s)
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
            ts.tstate = PyThreadState_New(CTX.interp)
            ts.tstate == C_NULL && error("PyThreadState_New failed")
        end
        s.tstate = ts.tstate
        PyEval_RestoreThread(s.tstate)
        s.attached = true
        return :root_owned
    catch
        acquired && Base.release(ts.sem)
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
end

function exit_region(task::Task, s::TaskState, token)
    token === :noop && return
    check_thread(task, s)
    if token === :detach
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        Base.release(s.sem::Base.Semaphore)
    elseif token === :root_owned
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        task.sticky = oldsticky
    elseif token === :root_borrowed
        @assert current_tstate() == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        task.sticky = oldsticky
    else
        error("invalid Python region token")
    end
    return
end

function enter_break(task::Task, s::TaskState)
    if s.tstate != C_NULL
        check_thread(task, s)
        !s.attached && return :noop
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        Base.release(s.sem::Base.Semaphore)
        return :restore
    end
    current_tstate() == C_NULL && return :noop

    ts = try
        start_session!(task, s)
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
        saved = PyEval_SaveThread()
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

function exit_break(task::Task, s::TaskState, token)
    token === :noop && return
    check_thread(task, s)
    Base.acquire(s.sem::Base.Semaphore)
    PyEval_RestoreThread(s.tstate)
    s.attached = true
    if token === :root_restore
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        reset!(s)
        Base.release(sem)
        task.sticky = oldsticky
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
        local task = current_task()
        local state = $TASK_STATE()
        local token = $enter_region(task, state)
        try
            $(esc(ex))
        finally
            $exit_region(task, state, token)
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
        local task = current_task()
        local state = $TASK_STATE()
        local token = $enter_break(task, state)
        try
            $(esc(ex))
        finally
            $exit_break(task, state, token)
        end
    end
end
