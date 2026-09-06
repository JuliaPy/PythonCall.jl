const THREAD_STATE_LOCK = ReentrantLock()
const THREAD_STATE_LOCK_PER_THREAD = OncePerThread{ReentrantLock}(ReentrantLock)
const THREAD_STATE = OncePerThread{Ptr{Cvoid}}(() -> PyThreadState_New(PyInterpreterState_Main()))

get_thread_state_lock() = CTX.is_free_threaded ? THREAD_STATE_LOCK_PER_THREAD() : THREAD_STATE_LOCK
get_thread_state() = THREAD_STATE()

"""
    @with_thread_state ex

Run the given expression `ex` with an attached CPython thread-state.

Limitations:
- This uses a `ReentrantLock` for co-operation with other Julia tasks so cannot be
  called in finalizers.
- The expression must not throw (this macro does not use try-finally).

Rather than using this macro directly, consider using the `_WithThreadState` functions
instead, like `PyObject_GetAttr_WithThreadState`.
"""
macro with_thread_state(ex)
    quote
        # task must be sticky to prevent the thread from changing during this block
        task = current_task()
        sticky = task.sticky
        task.sticky = true
        # acquire a re-entrant lock, so that no other task can set the thread state
        thelock = get_thread_state_lock()
        lock(thelock)
        # attach the python thread state. this blocks the thread until the thread state
        # is detached, hence the above lock, so that the blocking is co-operative with
        # other julia tasks, rather than just hanging the thread. since the lock is
        # re-entrant, the task might enter this locked area again while still locked
        # (that is, nesting this macro is allowed) so we use PyThreadState_Swap, which
        # will return NULL in the outermost invocation, and will return THREAD_STATE()
        # in all the innermost ones.
        tstate = get_thread_state()
        tstate0 = PyThreadState_Swap(tstate)
        # run the desired expression
        ans = $(esc(ex))
        # swap the threadstate back to its prior value
        tstate2 = PyThreadState_Swap(tstate0)
        # check the thread state didn't change
        @assert tstate2 == tstate
        # reset the task stickiness, so that a previously non-sticky task remains non-
        # sticky and can be migrated outside of this block
        task.sticky = sticky
        # unlock, to allow another task to call into python
        unlock(thelock)
        # return the result of the expression
        ans
    end
end

# create function wrappers for CPython functions with a _WithThreadState suffix that are
# called with an attached thread-state
for (name, (argtypes, rettype)) in CAPI_FUNC_SIGS
    args = [Symbol("x", i) for (i, _) in enumerate(argtypes)]
    @eval $(Symbol(name, "_WithThreadState"))($(args...)) = @with_thread_state ccall(POINTERS.$name, $rettype, ($(argtypes...),), $(args...))
end
