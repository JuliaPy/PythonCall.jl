for m in ["os", "sys", "pprint", "traceback", "numbers", "math", "collections", "collections.abc", "datetime", "fractions", "io", "types", "pdb"]
    j = Symbol(:py, replace(m, '.'=>""), :module)
    @eval const $j = pylazyobject(() -> pyimport($m))
end

"""
    pynewclass(name, bases=(), kwargs=nothing; attrs...)

Create a new Python class with the given name, base classes and attributes.
"""
pynewclass(name, bases=(), kwds=nothing; attrs...) =
    pytypesmodule.new_class(name, bases, kwds, pyjlfunction(ns -> (for (k,v) in attrs; ns[string(k)] = v; end)))
export pynewclass
