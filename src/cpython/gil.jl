"""
    with_gil(f, py, [c=true])

Compute `f()` with the GIL enabled.

This may need a `try-finally` block to ensure the GIL is released again. If you know that `f` cannot throw, pass `c=false` to avoid this overhead.
"""
@inline function with_gil(f, py::Context, c::Bool = true)
    if !py.is_embedded
        f()
    elseif c
        g = py.PyGILState_Ensure()
        try
            f()
        finally
            py.PyGILState_Release(g)
        end
    else
        g = py.Py_GILState_Ensure()
        r = f()
        py.Py_GILState_Release(g)
        r
    end
end

@inline (x::Func{:with_gil})(f, c::Bool=true) = with_gil(f, x.ctx, c)
