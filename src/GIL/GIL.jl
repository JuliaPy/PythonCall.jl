"""
    module PythonCall.GIL

Compatibility API for Python interaction regions.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).

!!! warning

    Multi-threading support is experimental and can change without notice.
"""
module GIL

using ..Region

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

Compute `f()` in a Python-heavy region and return its result.

PythonCall APIs already establish the required Python thread state automatically. This
compatibility function can amortize transitions across several operations.

See [`@lock`](@ref) for the macro form.

!!! warning

    This function is experimental. Its semantics may be changed without notice.
"""
function lock(f)
    token = Region.enter_region()
    try
        f()
    finally
        Region.exit_region(token)
    end
end

"""
    @lock expr

Compute `expr` in a Python-heavy region and return its result.

The macro equivalent of [`lock`](@ref).

!!! warning

    This macro is experimental. Its semantics may be changed without notice.
"""
macro lock(expr)
    quote
        token = $Region.enter_region()
        try
            $(esc(expr))
        finally
            $Region.exit_region(token)
        end
    end
end

"""
    unlock(f)

Temporarily relinquish Python-related resources, compute `f()`, restore them, and return
the result. Prefer [`PythonCall.@pyregionbreak`](@ref) in new code.

See [`@unlock`](@ref) for the macro form.

!!! warning

    This function is experimental. Its semantics may be changed without notice.
"""
function unlock(f)
    token = Region.enter_break()
    try
        f()
    finally
        Region.exit_break(token)
    end
end

"""
    @unlock expr

Temporarily relinquish Python-related resources, compute `expr`, restore them, and return
the result. Prefer [`PythonCall.@pyregionbreak`](@ref) in new code.

The macro equivalent of [`unlock`](@ref).

!!! warning

    This macro is experimental. Its semantics may be changed without notice.
"""
macro unlock(expr)
    quote
        token = $Region.enter_break()
        try
            $(esc(expr))
        finally
            $Region.exit_break(token)
        end
    end
end

end
