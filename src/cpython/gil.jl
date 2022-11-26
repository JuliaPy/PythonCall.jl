"""
    with_gil(f, [c=true])

Compute `f()` with the GIL enabled.

This may need a `try-finally` block to ensure the GIL is released again. If you know that `f` cannot throw, pass `c=false` to avoid this overhead.
"""
@inline function with_gil(f, c::Bool = true)
    if !CTX[].is_embedded
        f()
    elseif c
        g = PyGILState_Ensure()
        try
            f()
        finally
            PyGILState_Release(g)
        end
    else
        g = PyGILState_Ensure()
        r = f()
        PyGILState_Release(g)
        r
    end
end
