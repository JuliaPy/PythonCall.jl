const PyThreadStatePtr = Ptr{Cvoid}

mutable struct TaskState
    # CPython state used by this task for the duration of its outermost region.
    tstate::PyThreadStatePtr
    # Semaphore protecting `tstate`; `nothing` when the task is outside a session.
    sem::Union{Nothing,Base.Semaphore}
    # Whether `tstate` belongs to the Python thread which called into Julia.
    borrowed::Bool
    # Whether `tstate` is currently attached to the task's pinned OS thread.
    attached::Bool
    # Julia thread to which the task is pinned while its session is active.
    tid::Int
    # Stickiness to restore when the task's outermost region or break finishes.
    oldsticky::Bool
end
TaskState() = TaskState(C_NULL, nothing, false, false, 0, false)

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
# state attached; that state is borrowed instead. Borrowed states must not acquire
# the semaphore while holding the GIL: another task can own the semaphore while
# waiting for the GIL, so doing so would invert the lock order and deadlock.
#
# Nested regions merely return `:noop`. For a PythonCall-owned state, a break saves
# and detaches the state, clears the thread cache, and releases the semaphore so
# other sticky tasks can use this Julia thread while the original task yields. On
# return it reacquires the semaphore before restoring both the state and cache. A
# borrowed callback state instead detaches and restores without touching the cache
# or semaphore. Consequently the cache is populated exactly while a task owns the
# semaphore with Python attached. This invariant makes its lock-free fast path safe.
#
# An attached CPython state is also a still cheaper fast path for @pyregion itself.
# Given that PythonCall is the only Julia-side entry to Python, confirmed ownership
# means either that an enclosing region already did the bookkeeping or that Python
# called into Julia and lent us its state. In either case a nested region has no
# transition to perform. Conversely, a region inside a break observes no attached
# state and takes the full path below, reacquiring and restoring its task state.

current_tstate() = PyThreadState_GetUnchecked()
function tstate_attached()
    # Before Python 3.12, _PyThreadState_UncheckedGet can return the thread's
    # registered state even after PyEval_SaveThread released the GIL. On a GIL
    # build, PyGILState_Check is the API which proves that using Python is safe.
    return CTX.is_free_threaded ? current_tstate() != C_NULL : !iszero(PyGILState_Check())
end
has_tstate() = CTX.is_initialized && tstate_attached()

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
    s.borrowed = false
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
        if s.borrowed
            PyEval_RestoreThread(s.tstate)
            s.attached = true
            return :detach_borrowed
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
    try
        if has_tstate()
            # Calls originating in Python borrow that thread's state without taking
            # the Julia-thread semaphore. The GIL already protects the state here.
            s.tstate = current_tstate()
            s.borrowed = true
            s.attached = true
            return :root_borrowed
        end
        acquired = false
        Base.acquire(ts.sem)
        acquired = true
        try
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
            rethrow()
        end
    catch
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
    if token === :detach_borrowed
        saved = PyEval_SaveThread()
        @assert saved == s.tstate
        s.attached = false
    elseif token === :detach
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
        oldsticky = s.oldsticky
        reset!(s)
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
        s.borrowed && return :restore_borrowed
        clear_task!(THREAD_STATE())
        Base.release(s.sem::Base.Semaphore)
        return :restore
    end
    # Outside a region, a break matters only in a Python-originated callback. It
    # borrows and detaches that state so yielding Julia code cannot retain Python.
    !has_tstate() && return :noop

    try
        start_session!(task, s)
    catch
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
    try
        if !has_tstate()
            task.sticky = s.oldsticky
            reset!(s)
            return :noop
        end
        current = current_tstate()
        s.tstate = current
        s.borrowed = true
        saved = PyEval_SaveThread()
        @assert saved == current
        s.attached = false
        return :root_restore
    catch
        task.sticky = s.oldsticky
        reset!(s)
        rethrow()
    end
end

function exit_break(task::Task, s::TaskState, token)
    token === :noop && return
    check_thread(task, s)
    if token === :root_restore
        # Restore a state borrowed from a Python callback directly. Acquiring the
        # Julia-thread semaphore here would invert its lock order with the GIL.
        PyEval_RestoreThread(s.tstate)
        oldsticky = s.oldsticky
        reset!(s)
        task.sticky = oldsticky
        return
    elseif token === :restore_borrowed
        PyEval_RestoreThread(s.tstate)
        s.attached = true
        return
    elseif token !== :restore
        error("invalid Python region-break token")
    end
    Base.acquire(s.sem::Base.Semaphore)
    PyEval_RestoreThread(s.tstate)
    s.attached = true
    set_task!(THREAD_STATE(), task, s)
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
        if $has_tstate()
            # The ownership check proves that this region is nested or
            # Python-originated; avoid both Julia task- and thread-local lookups.
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
