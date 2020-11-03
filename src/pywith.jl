"""
    pywith(f, o)

Simulates the Python `with` statement, calling `f(x)` where `x=o.__enter__()`.
"""
function pywith(f, o::AbstractPyObject)
    enter = pytype(o).__enter__
    exit = pytype(o).__exit__
    value = enter(o)
    hit_except = false
    try
        return f(value)
    catch err
        if err isa PythonRuntimeError
            hit_except = true
            if pytruth(exit(o, err.t, err.v, err.b))
                rethrow()
            end
        else
            rethrow()
        end
    finally
        if !hit_except
            exit(o, pynone, pynone, pynone)
        end
    end
end
export pywith
