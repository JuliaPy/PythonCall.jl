const PyThreadStatePtr = Ptr{Cvoid}

mutable struct TaskState
    # CPython state used by this task for the duration of its outermost region.
    tstate::PyThreadStatePtr
    # Semaphore protecting `tstate`; `nothing` when the task is outside a session.
    sem::Union{Nothing,Base.Semaphore}
    # Whether `tstate` is currently attached to the task's pinned OS thread.
    attached::Bool
    # Julia thread to which the task is pinned while its session is active.
    tid::Int
    # Stickiness to restore when the task's outermost region or break finishes.
    oldsticky::Bool
end
TaskState() = TaskState(C_NULL, nothing, false, 0, false)

mutable struct ThreadState
    # Serializes tasks which use the persistent CPython state on this Julia thread.
    sem::Base.Semaphore
    # PythonCall-owned CPython state for this Julia thread, allocated lazily.
    tstate::PyThreadStatePtr
    # Task which currently holds `sem` with a CPython state attached, if any.
    task::Union{Nothing,Task}
    # State belonging to `task`; cached so nested regions need no task-local lookup.
    task_state::Union{Nothing,TaskState}
end
ThreadState() = ThreadState(Base.Semaphore(1), C_NULL, nothing, nothing)

const THREAD_STATE = Utils.OncePerThread{ThreadState}(ThreadState)
const TASK_STATE = Utils.OncePerTask{TaskState}(TaskState)

# A region session starts at the outermost region entered by a Julia task and ends
# when that region exits. The task is made sticky for the whole session: CPython
# thread states must be restored on the same OS thread on which they were saved.
# The corresponding ThreadState semaphore gives the task exclusive use of that
# thread's persistent PyThreadState. A Python-originated call already has a CPython
# state attached; that state is borrowed instead, but is protected by the same
# semaphore so PythonCall's accounting remains identical.
#
# Nested regions merely return `:noop`. A break saves/detaches the CPython state,
# clears the thread cache, and releases the semaphore, allowing other sticky tasks
# to use this Julia thread while the original task yields. On return it reacquires
# the semaphore before restoring both the CPython state and cache. Consequently the
# cache is populated exactly while a task owns the semaphore with Python attached.
# This invariant is what makes its lock-free fast path safe.
#
# An attached CPython state is also a still cheaper fast path for @pyregion itself.
# Given that PythonCall is the only Julia-side entry to Python, an attached state
# means either that an enclosing region already did the bookkeeping or that Python
# called into Julia and lent us its state. In either case a nested region has no
# transition to perform. Conversely, a region inside a break observes no attached
# state and takes the full path below, reacquiring and restoring its task state.

current_tstate() = PyThreadState_GetUnchecked()
has_tstate() = CTX.is_initialized && current_tstate() != C_NULL

# An attached task is pinned to this Julia thread and owns its semaphore, so the
# thread-local cache cannot change underneath it.  This makes nested regions avoid
# the considerably more expensive OncePerTask lookup.  The cache is cleared before
# releasing the semaphore and restored after reacquiring it around a region break.
function current_task_state(task::Task)
    ts = THREAD_STATE()
    return ts.task === task ? ts.task_state::TaskState : TASK_STATE()
end

function set_task!(ts::ThreadState, task::Task, s::TaskState)
    ts.task = task
    ts.task_state = s
    return
end

function clear_task!(ts::ThreadState)
    ts.task = nothing
    ts.task_state = nothing
    return
end

function reset!(s::TaskState)
    # `oldsticky` is deliberately left alone: start_session! overwrites it before
    # the TaskState becomes observable as an active session again.
    s.tstate = C_NULL
    s.sem = nothing
    s.attached = false
    s.tid = 0
    return
end

function start_session!(task::Task, s::TaskState)
    # Pin before recording the thread and looking up its ThreadState. Once sticky,
    # cooperative scheduling cannot resume this task on another Julia thread.
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
        # An existing session is either already attached (ordinary nesting), or
        # temporarily detached because this region is nested inside a break.
        check_thread(task, s)
        if s.attached
            return :noop
        end
        Base.acquire(s.sem::Base.Semaphore)
        try
            PyEval_RestoreThread(s.tstate)
            s.attached = true
            set_task!(THREAD_STATE(), task, s)
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
            # Calls originating in Python must return with its state still attached.
            s.tstate = current
            s.attached = true
            set_task!(ts, task, s)
            return :root_borrowed
        end
        if ts.tstate == C_NULL
            ts.tstate = PyThreadState_New(CTX.interp)
            ts.tstate == C_NULL && error("PyThreadState_New failed")
        end
        s.tstate = ts.tstate
        PyEval_RestoreThread(s.tstate)
        s.attached = true
        set_task!(ts, task, s)
        return :root_owned
    catch
        acquired && Base.release(ts.sem)
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
end

function exit_region(task::Task, s::TaskState, token)
    # Tokens encode precisely which work enter_region performed, so every path
    # reverses only its own transition and nested :noop regions do no attach work.
    token === :noop && return
    check_thread(task, s)
    if token === :detach
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        clear_task!(THREAD_STATE())
        Base.release(s.sem::Base.Semaphore)
    elseif token === :root_owned
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        clear_task!(THREAD_STATE())
        reset!(s)
        Base.release(sem)
        task.sticky = oldsticky
    elseif token === :root_borrowed
        @assert current_tstate() == s.tstate
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        clear_task!(THREAD_STATE())
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
        # Only the outermost break detaches. Further breaks while detached are no-ops.
        check_thread(task, s)
        !s.attached && return :noop
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
        clear_task!(THREAD_STATE())
        Base.release(s.sem::Base.Semaphore)
        return :restore
    end
    # Outside a region, a break matters only in a Python-originated callback. It
    # borrows and detaches that state so yielding Julia code cannot retain Python.
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
    set_task!(THREAD_STATE(), task, s)
    if token === :root_restore
        # The borrowed Python state stays attached for the caller, but the temporary
        # Julia session and its cache ownership end here.
        sem, oldsticky = s.sem::Base.Semaphore, s.oldsticky
        clear_task!(THREAD_STATE())
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
        if $current_tstate() != C_NULL
            # The CPython TLS lookup is enough to prove that this region is nested
            # or Python-originated; avoid both Julia task- and thread-local lookups.
            $(esc(ex))
        else
            local task = current_task()
            local state = $current_task_state(task)
            local token = $enter_region(task, state)
            try
                $(esc(ex))
            finally
                $exit_region(task, state, token)
            end
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
        local state = $current_task_state(task)
        local token = $enter_break(task, state)
        try
            $(esc(ex))
        finally
            $exit_break(task, state, token)
        end
    end
end
