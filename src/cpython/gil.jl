"""
    with_gil(f, ctx, [c=true])

Compute `f()` with the GIL enabled.

This may need a `try-finally` block to ensure the GIL is released again. If you know that `f` cannot throw, pass `c=false` to avoid this overhead.
"""
@inline function with_gil(f, ctx::Context, c::Bool = true)
    if !ctx.is_embedded
        f()
    elseif c
        g = ctx.PyGILState_Ensure()
        try
            f()
        finally
            ctx.PyGILState_Release(g)
        end
    else
        g = ctx.Py_GILState_Ensure()
        r = f()
        ctx.Py_GILState_Release(g)
        r
    end
end

@inline (x::Func{:with_gil})(f, c::Bool=true) = with_gil(f, x.ctx, c)
