"""
    TaskState

When a `Task` acquires the GIL, save the GIL state and the stickiness of the
`Task` since we will force the `Task` to be sticky. We need to restore the GIL
state on release of the GIL via `C.PyGILState_Release`.
"""
struct TaskState
    task::Task
    sticky::Bool # original stickiness of the task
    state::C.PyGILState_STATE
end

"""
    TaskStack

For each thread the `TaskStack` maintains a first-in-last-out list of tasks
as well as the GIL state and their stickiness upon entering the stack. This
forces tasks to unlock the GIL in the reverse order of which they locked it.
"""
struct TaskStack
    stack::Vector{TaskState}
    count::IdDict{Task,Int}
    condvar::Threads.Condition
    function TaskStack()
        return new(TaskState[], IdDict{Task,Int}(), Threads.Condition())
    end
end
function Base.last(task_stack::TaskStack)::Task
    return last(task_stack.stack).task
end
function Base.push!(task_stack::TaskStack, task::Task)
    original_sticky = task.sticky
    # The task should not migrate threads while acquiring or holding the GIL
    task.sticky = true
    gil_state = C.PyGILState_Ensure()

    # Save the stickiness and state for when we release
    state = TaskState(task, original_sticky, gil_state)
    push!(task_stack.stack, state)

    # Increment the count for this task
    count = get(task_stack.count, task, 0)
    task_stack.count[task] = count + 1

    return task_stack
end
function Base.pop!(task_stack::TaskStack)::Task
    state = pop!(task_stack.stack)
    task = state.task
    sticky = state.sticky
    gil_state = state.state

    # Decrement the count for this task
    count = task_stack.count[task] - 1
    if count == 0
        # If 0, remove it from the key set
        pop!(task_stack.count, task)
    else
        task_stack.count[task] = count
    end

    C.PyGILState_Release(gil_state)

    # Restore sticky state after releasing the GIL
    task.sticky = sticky

    Base.lock(task_stack.condvar) do
        notify(task_stack.condvar)
    end

    return task
end
Base.in(task::Task, task_stack::TaskStack) = haskey(task_stack.count)
Base.isempty(task_stack::TaskStack) = isempty(task_stack.stack)

if !isdefined(Base, :OncePerThread)

    const PerThreadLock = Base.ThreadSynchronizer()

    # OncePerThread is implemented in full in Julia 1.12
    # This implementation is meant for compatibility with Julia 1.10 and 1.11.
    # Using Julia 1.12 is recommended.
    mutable struct OncePerThread{T,F} <: Function
        @atomic xs::Dict{Int, T} # values
        @atomic ss::Dict{Int, UInt8} # states: 0=initial, 1=hasrun, 2=error, 3==concurrent
        const initializer::F
        function OncePerThread{T,F}(initializer::F) where {T,F}
            nt = Threads.maxthreadid()
            return new{T,F}(Dict{Int,T}(), Dict{Int,UInt8}(), initializer)
        end
    end
    OncePerThread{T}(initializer::Type{U}) where {T, U} = OncePerThread{T,Type{U}}(initializer)
    (once::OncePerThread{T,F})() where {T,F} = once[Threads.threadid()]
    function Base.getindex(once::OncePerThread, tid::Integer)
        tid = Int(tid)
        ss = @atomic :acquire once.ss
        xs = @atomic :monotonic once.xs

        if haskey(ss, tid) && ss[tid] == 1
            return xs[tid]
        end

        Base.lock(PerThreadLock)
        try
            state = get(ss, tid, 0)
            if state == 0
                xs[tid] = once.initializer()
                ss[tid] = 1
            end
        finally
            Base.unlock(PerThreadLock)
        end
        return xs[tid]
    end
end

"""
    GlobalInterpreterLock

Provides a thread aware reentrant lock around Python's interpreter lock that
ensures that `Task`s acquiring the lock stay on the same thread.
"""
struct GlobalInterpreterLock <: Base.AbstractLock
    lock_owners::OncePerThread{TaskStack}
    function GlobalInterpreterLock()
        return new(OncePerThread{TaskStack}(TaskStack))
    end
end
function Base.lock(gil::GlobalInterpreterLock)
    push!(gil.lock_owners(), current_task())
    return nothing
end
function Base.unlock(gil::GlobalInterpreterLock)
    lock_owner::TaskStack = gil.lock_owners()
    last_owner::Task = if isempty(lock_owner)
        current_task()
    else
        last(lock_owner)
    end
    while last_owner != current_task()
        if istaskdone(last_owner) && !isempty(lock_owner)
            # Last owner is done and unable to unlock the GIL
            pop!(lock_owner)
            error("Unlock from the wrong task. The Task that owned the GIL is done and did not unlock the GIL: $(last_owner)")
        else
            # This task does not own the GIL. Wait to unlock the GIL until
            # another task successfully unlocks the GIL.
            wait(lock_owner.condvar)
        end
        last_owner = if isempty(lock_owner)
            current_task()
        else
            last(lock_owner)
        end
    end
    if isempty(lock_owner)
        error("Unlock from wrong task: $(current_task). No tasks on this thread own the lock.")
    else
        task = pop!(lock_owner)
    end
    @assert task == current_task()
    return nothing
end
function Base.islocked(gil::GlobalInterpreterLock)
    return any(!isempty(gil.lock_owners[thread_index]) for thread_index in 1:Threads.maxthreadid())
end
function haslock(gil::GlobalInterpreterLock, task::Task)
    lock_owner::TaskStack = gil.lock_owners()
    if isempty(lock_owner)
        return false
    end
    return last(lock_owner)::Task == task
end

const _GIL = GlobalInterpreterLock()
