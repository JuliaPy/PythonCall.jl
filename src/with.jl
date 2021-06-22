"""
    pywith(f, o, d=nothing)

Equivalent to `with o as x: f(x)` in Python, where `x` is a `PyObject`.

On success, the value of `f(x)` is returned.

If an exception occurs but is suppressed then `d` is returned.
"""
function pywith(f, o, d = nothing)
    o = Py(o)
    t = pytype(o)
    exit = t.__exit__
    value = t.__enter__(o)
    exited = false
    try
        f(value)
    catch exc
        if exc isa PyException
            exited = true
            if pytruth(exit(o, exc.t, exc.v, exc.b))
                rethrow()
            else
                d
            end
        else
            rethrow()
        end
    finally
        exited || exit(o, pybuiltins.None, pybuiltins.None, pybuiltins.None)
    end
end
export pywith
