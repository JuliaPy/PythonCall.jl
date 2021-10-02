# Builtin functions not covered elsewhere

"""
    pyprint(...)

Equivalent to `print(...)` in Python.
"""
pyprint(args...; kwargs...) = (pydel!(pybuiltins.print(args...; kwargs...)); nothing)
export pyprint

"""
    pyhelp(x)

Equivalent to `help(x)` in Python.
"""
pyhelp(x) = (pydel!(pybuiltins.help(x)); nothing)
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
