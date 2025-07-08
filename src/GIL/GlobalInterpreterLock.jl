struct TaskState
    task::Task
    sticky::Bool # original stickiness of the task
    state::C.PyGILState_STATE
end

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
        task_stack[task] = count
    end

    C.PyGILState_Release(gil_state)

    # Restore sticky state after releasing the GIL
    task.sticky = sticky

    Base.lock(task_stack.condvar) do
        notify(task_stack.condvar)
    end

    return task
end
Base.isempty(task_stack::TaskStack) = isempty(task_stack.stack)

if !isdefined(Base, :OncePerThread)

    # OncePerThread is implemented in full in Julia 1.12
    # This implementation is meant for compatibility with Julia 1.10 and 1.11
    # and only supports a static number of threads. Use Julia 1.12 for dynamic
    # thread usage.
    mutable struct OncePerThread{T,F} <: Function
        @atomic xs::Vector{T} # values
        @atomic ss::Vector{UInt8} # states: 0=initial, 1=hasrun, 2=error, 3==concurrent
        const initializer::F
        function OncePerThread{T,F}(initializer::F) where {T,F}
            nt = Threads.maxthreadid()
            return new{T,F}(Vector{T}(undef, nt), zeros(UInt8, nt), initializer)
        end
    end
    OncePerThread{T}(initializer::Type{U}) where {T, U} = OncePerThread{T,Type{U}}(initializer)
    (once::OncePerThread{T,F})() where {T,F} = once[Threads.threadid()]
    function Base.getindex(once::OncePerThread, tid::Integer)
        tid = Threads.threadid()
        ss = @atomic :acquire once.ss
        xs = @atomic :monotonic once.xs
        if checkbounds(Bool, xs, tid)
            if ss[tid] == 0
                xs[tid] = once.initializer()
                ss[tid] = 1
            end
            return xs[tid]
        else
            throw(ErrorException("Thread id $tid is out of bounds as initially allocated. Use Julia 1.12 for dynamic thread usage."))
        end
    end

end

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
    while last(lock_owner) != current_task()
        wait(lock_owner.condvar)
    end
    task = pop!(lock_owner)
    @assert task == current_task()
    return nothing
end
function Base.islocked(gil::GlobalInterpreterLock)
    # TODO: handle Julia 1.10 and 1.11 case when have not allocated up to maxthreadid
    return any(!isempty(gil.lock_owners[thread_index]) for thread_index in 1:Threads.maxthreadid())
end

const _GIL = GlobalInterpreterLock()
