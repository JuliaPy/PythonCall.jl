"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).
"""
module GIL

using ..C: C

# Ensure that only one Julia task tries to acquire the Python GIL.
# Avoid the potential issue that a task could miscompute whether
# it actually has the GIL simply because a different task that ran
# on the same thread that once had the GIL.
# https://github.com/JuliaPy/PythonCall.jl/issues/627
const _jl_gil_lock = ReentrantLock()

"""
    hasgil()

Returns `true` if the current thread has the GIL or `false` otherwise.
"""
hasgil() = C.PyGILState_Check() == Cint(1)

"""
    lock(f; exclusive=true)

Lock the GIL, compute `f()`, unlock the GIL, then return the result of `f()`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`unlock`](@ref) the GIL before using this function.

Setting the `exclusive` keyword to `false` allows more than one Julia `Task`
to attempt to acquire the GIL. This may be useful if one Julia `Task` calls
code which releases the GIL, in which case another Julia task could acquire
it. However, this is probably not a sound approach.

See [`@lock`](@ref) for the macro form.
"""
function lock(f; exclusive=true)
    exclusive && Base.lock(_jl_gil_lock)
    try
        state = C.PyGILState_Ensure()
        try
            f()
        finally
            C.PyGILState_Release(state)
        end
    finally
        exclusive && Base.unlock(_jl_gil_lock)
    end
end

"""
    @lock [exclusive=true] expr

Lock the GIL, compute `expr`, unlock the GIL, then return the result of `expr`.

Use this to run Python code from threads that do not currently hold the GIL, such as new
threads. Since the main Julia thread holds the GIL by default, you will need to
[`@unlock`](@ref) the GIL before using this function.

Setting the `exclusive` parameter to `false` allows more than one Julia `Task`
to attempt to acquire the GIL. This may be useful if one Julia `Task` calls
code which releases the GIL, in which case another Julia task could acquire
it. However, this is probably not a sound approach.

The macro equivalent of [`lock`](@ref).
"""
macro lock(expr)
    quote
        Base.lock(_jl_gil_lock)
        try
            state = C.PyGILState_Ensure()
            try
                $(esc(expr))
            finally
                C.PyGILState_Release(state)
            end
        finally
            Base.unlock(_jl_gil_lock)
        end
    end
end

macro lock(parameter, expr)
    parameter.head == :(=) &&
        parameter.args[1] == :exclusive ||
        throw(ArgumentError("The only accepted parameter to @lock is `exclusive`."))

    do_lock = esc(parameter.args[2])
    quote
        $do_lock && Base.lock(_jl_gil_lock)
        try
            state = C.PyGILState_Ensure()
            try
                $(esc(expr))
            finally
                C.PyGILState_Release(state)
            end
        finally
            $do_lock && Base.unlock(_jl_gil_lock)
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
    _locked = Base.islocked(_jl_gil_lock)
    _locked && Base.unlock(_jl_gil_lock)
    try
        state = C.PyEval_SaveThread()
        try
            f()
        finally
            C.PyEval_RestoreThread(state)
        end
    finally
        _locked && Base.lock(_jl_gil_lock)
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
        _locked = Base.islocked(_jl_gil_lock)
        _locked && Base.unlock(_jl_gil_lock)
        try
            state = C.PyEval_SaveThread()
            try
                $(esc(expr))
            finally
                C.PyEval_RestoreThread(state)
            end
        finally
            _locked && Base.lock(_jl_gil_lock)
        end
    end
end

#=
# Disable this for now since holding this lock will disable finalizers
# If the main thread already has the GIL, we should lock _jl_gil_lock.
function __init__()
    if hasgil()
        Base.lock(_jl_gil_lock)
    end
end
=#

include("GlobalInterpreterLock.jl")

end
