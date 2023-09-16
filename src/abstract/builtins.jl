# Builtin functions not covered elsewhere

"""
    pyprint(...)

Equivalent to `print(...)` in Python.
"""
pyprint(args...; kwargs...) = (pydel!(pybuiltins.print(args...; kwargs...)); nothing)
export pyprint

function _pyhelp(args...)
    pyisnone(pybuiltins.help) && error("Python help is not available")
    pydel!(pybuiltins.help(args...))
    nothing
end
"""
    pyhelp([x])

Equivalent to `help(x)` in Python.
"""
pyhelp() = _pyhelp()
pyhelp(x) = _pyhelp(x)
export pyhelp

"""
    pyall(x)

Equivalent to `all(x)` in Python.
"""
function pyall(x)
    y = pybuiltins.all(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pyall

"""
    pyany(x)

Equivalent to `any(x)` in Python.
"""
function pyany(x)
    y = pybuiltins.any(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pyany

"""
    pycallable(x)

Equivalent to `callable(x)` in Python.
"""
function pycallable(x)
    y = pybuiltins.callable(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pycallable

"""
    pycompile(...)

Equivalent to `compile(...)` in Python.
"""
pycompile(args...; kwargs...) = pybuiltins.compile(args...; kwargs...)
export pycompile
