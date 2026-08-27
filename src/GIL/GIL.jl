"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).

!!! warning

    Multi-threading support is experimental and can change without notice.
"""
module GIL

using ..C: C

if Base.VERSION ≥ v"1.11"
    eval(
        Expr(
            :public,
            :lock,
            Symbol("@lock"),
            :unlock,
            Symbol("@unlock"),
        ),
    )
end


"""
    lock(f; sticky::Bool=true)

Lock the GIL, compute `f()`, unlock the GIL, then return the result of `f()`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`unlock`](@ref) the GIL before using this function.

See [`@lock`](@ref) for the macro form.

The optional boolean argument `sticky` controls whether the calling task is marked as
sticky.  The default behavior is to mark the current task as sticky, as Julia's scheduler
may migrate non-sticky tasks to another OS thread; since the GIL is specific to the OS
thread, holding the GIL on a non-sticky thread is potentially dangerous and can lead to
undefined behavior like deadlocks and segfaults.

!!! warning

    This function is experimental. Its semantics may be changed without notice.
"""
function lock(f; sticky::Bool=true)
    state = C.PyGILState_Ensure()
    was_sticky = current_task().sticky
    try
        if sticky
            current_task().sticky = true
        end
        f()
    finally
        C.PyGILState_Release(state)
        current_task().sticky = was_sticky
    end
end

"""
    @lock [sticky::Bool=true] expr

Lock the GIL, compute `expr`, unlock the GIL, then return the result of `expr`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`@unlock`](@ref) the GIL before using this function.

The macro equivalent of [`lock`](@ref).

The optional boolean argument `sticky` controls whether the calling task is marked as
sticky.  The default behavior is to mark the current task as sticky, as Julia's scheduler
may migrate non-sticky tasks to another OS thread; since the GIL is specific to the OS
thread, holding the GIL on a non-sticky thread is potentially dangerous and can lead to
undefined behavior like deadlocks and segfaults.

!!! warning

    This macro is experimental. Its semantics may be changed without notice.
"""
macro lock(expr)
    _lock_impl(true, expr)
end

macro lock(sticky, expr)
    _lock_impl(sticky, expr)
end

function _lock_impl(sticky::Bool, expr)
    quote
        state = C.PyGILState_Ensure()
        was_sticky = current_task().sticky
        try
            if $sticky
                current_task().sticky = true
            end
            $(esc(expr))
        finally
            C.PyGILState_Release(state)
            current_task().sticky = was_sticky
        end
    end
end

"""
    unlock(f)

Unlock the GIL, compute `f()`, re-lock the GIL, then return the result of `f()`.

Use this to run non-Python code with the GIL unlocked, so allowing another thread to run
Python code. That other thread can be a Julia thread, which must lock the GIL using
[`lock`](@ref).

See [`@unlock`](@ref) for the macro form.

!!! warning

    This function is experimental. Its semantics may be changed without notice.
"""
function unlock(f)
    state = C.PyEval_SaveThread()
    try
        f()
    finally
        C.PyEval_RestoreThread(state)
    end
end

"""
    @unlock expr

Unlock the GIL, compute `expr`, re-lock the GIL, then return the result of `expr`.

Use this to run non-Python code with the GIL unlocked, so allowing another thread to run
Python code. That other thread can be a Julia thread, which must lock the GIL using
[`@lock`](@ref).

The macro equivalent of [`unlock`](@ref).

!!! warning

    This macro is experimental. Its semantics may be changed without notice.
"""
macro unlock(expr)
    quote
        state = C.PyEval_SaveThread()
        try
            $(esc(expr))
        finally
            C.PyEval_RestoreThread(state)
        end
    end
end

end
