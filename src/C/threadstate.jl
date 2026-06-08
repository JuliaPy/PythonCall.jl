const THREAD_STATE_LOCK = ReentrantLock()
const THREAD_STATE_LOCK_PER_THREAD = OncePerThread{ReentrantLock}(ReentrantLock)
const THREAD_STATE = OncePerThread{Ptr{Cvoid}}(() -> PyThreadState_New(PyInterpreterState_Main()))

get_thread_state_lock() = CTX.is_free_threaded ? THREAD_STATE_LOCK_PER_THREAD() : THREAD_STATE_LOCK

macro with_thread_state(ex)
    quote
        # acquire a re-entrant lock, so that no other task can set the thread state
        thelock = get_thread_state_lock()
        lock(thelock)
        # task must be sticky to prevent the thread from changing during this block
        task = current_task()
        sticky = task.sticky
        task.sticky = true
        # attach the python thread state. this blocks the thread until the thread state
        # is detached, hence the above lock, so that the blocking is co-operative with
        # other julia tasks, rather than just hanging the thread. since the lock is
        # re-entrant, the task might enter this locked area again while still locked
        # (that is, nesting this macro is allowed) so we use PyThreadState_Swap, which
        # will return NULL in the outermost invocation, and will return THREAD_STATE()
        # in all the innermost ones.
        tstate = THREAD_STATE()
        tstate0 = PyThreadState_Swap(tstate)
        # to check it didn't change
        tid1 = Threads.threadid()
        # run the desired expression
        ans = $(esc(ex))
        # check the thread didn't change. it shouldn't as the task is sticky.
        tid2 = Threads.threadid()
        @assert tid2 == tid1
        # swap the threadstate back to its prior value
        tstate2 = PyThreadState_Swap(tstate0)
        # check the thread state didn't change
        @assert tstate2 == tstate
        # reset the task stickiness
        task.sticky = sticky
        # unlock, to allow another task to call into python
        unlock(thelock)
        # return the result of the expression
        ans
    end
end
