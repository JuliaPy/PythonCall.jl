"""
    with_gil(f, [c=true])

Compute `f()` with the GIL enabled.

This may need a `try-finally` block to ensure the GIL is released again. If you know that `f` cannot throw, pass `c=false` to avoid this overhead.
"""
function with_gil(f, c::Bool=true)
    if !CONFIG.isembedded
        f()
    elseif c
        g = C.PyGILState_Ensure()
        try
            f()
        finally
            C.PyGILState_Release(g)
        end
    else
        g = C.PyGILState_Ensure()
        r = f()
        C.PyGILState_Release(g)
        r
    end
end
