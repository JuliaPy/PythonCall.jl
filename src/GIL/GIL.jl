"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`release`](@ref) and [`@release`](@ref).
"""
module GIL

using ..C: C

"""
    lock(f)

Acquire the GIL, compute `f()`, release the GIL, then return the result of `f()`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`release`](@ref) the GIL before using this function.

See [`@lock`](@ref) for the macro form.
"""
function lock(f)
    state = C.PyGILState_Ensure()
    try
        f()
    finally
        C.PyGILState_Release(state)
    end
end

"""
    @lock expr

Acquire the GIL, compute `expr`, release the GIL, then return the result of `expr`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`@release`](@ref) the GIL before using this function.

The macro equivalent of [`lock`](@ref).
"""
macro lock(expr)
    quote
        state = C.PyGILState_Ensure()
        try
            $(esc(expr))
        finally
            C.PyGILState_Release(state)
        end
    end
end

"""
    release(f)

Release the GIL, compute `f()`, re-acquire the GIL, then return the result of `f()`.

Use this to run non-Python code with the GIL released, so allowing another thread to run
Python code. That other thread can be a Julia thread, which must acquire the GIL using
[`lock`](@ref).

See [`@release`](@ref) for the macro form.
"""
function release(f)
    state = C.PyEval_SaveThread()
    try
        f()
    finally
        C.PyEval_RestoreThread(state)
    end
end

"""
    @release expr

Release the GIL, compute `expr`, re-acquire the GIL, then return the result of `expr`.

Use this to run non-Python code with the GIL released, so allowing another thread to run
Python code. That other thread can be a Julia thread, which must acquire the GIL using
[`@lock`](@ref).

The macro equivalent of [`release`](@ref).
"""
macro release(expr)
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
