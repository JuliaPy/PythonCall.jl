"""
    with_gil(f, ctx, [c=true])

Compute `f()` with the GIL enabled.

This may need a `try-finally` block to ensure the GIL is released again. If you know that `f` cannot throw, pass `c=false` to avoid this overhead.
"""
@inline function with_gil(f, ctx::Context, c::Bool = true)
    if !ctx.is_embedded
        f()
    elseif c
        g = ccall(ctx.pointers.PyGILState_Ensure, Cint, ())
        try
            f()
        finally
            ccall(ctx.pointers.PyGILState_Release, Cvoid, (Cint,), g)
        end
    else
        g = ccall(ctx.pointers.PyGILState_Ensure, Cint, ())
        r = f()
        ccall(ctx.pointers.PyGILState_Release, Cvoid, (Cint,), g)
        r
    end
end
