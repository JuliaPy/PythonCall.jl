"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).
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
    lock(f)

Lock the GIL, compute `f()`, unlock the GIL, then return the result of `f()`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`unlock`](@ref) the GIL before using this function.

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

Lock the GIL, compute `expr`, unlock the GIL, then return the result of `expr`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`@unlock`](@ref) the GIL before using this function.

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
    unlock(f)

Unlock the GIL, compute `f()`, re-lock the GIL, then return the result of `f()`.

Use this to run non-Python code with the GIL unlocked, so allowing another thread to run
Python code. That other thread can be a Julia thread, which must lock the GIL using
[`lock`](@ref).

See [`@unlock`](@ref) for the macro form.
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
