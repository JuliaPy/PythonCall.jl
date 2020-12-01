for m in ["os", "sys", "pprint", "traceback", "numbers", "math", "collections", "collections.abc", "datetime", "fractions", "io", "types"]
    j = Symbol(:py, replace(m, '.'=>""), :module)
    @eval const $j = pylazyobject(() -> pyimport($m))
end

"""
    pynewclass(name, bases=(), kwargs=nothing; attrs...)

Create a new Python class with the given name, base classes and attributes.
"""
pynewclass(name, bases=(), kwargs=nothing; attrs...) =
    py"""
    def f(c):
        c.update($(pydict_fromstringiter(attrs)))
        return c
    r = $(pytypesmodule).new_class($name, $bases, $kwargs, f)
    """cl["r"]
export pynewclass
