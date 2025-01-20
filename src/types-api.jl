"""
    Py(x)

Convert `x` to a Python object, of type `Py`.

Conversion happens according to [these rules](@ref jl2py-conversion).

Such an object supports attribute access (`obj.attr`), indexing (`obj[idx]`), calling
(`obj(arg1, arg2)`), iteration (`for x in obj`), arithmetic (`obj + obj2`) and comparison
(`obj > obj2`), among other things. These operations convert all their arguments to `Py` and
return `Py`.
"""
mutable struct Py
    ptr::Ptr{Cvoid}
    Py(::Val{:new}, ptr::Ptr) = finalizer(Internals.Core.py_finalizer, new(Ptr{Cvoid}(ptr)))
end

"""
    PyException(x)

Wraps the Python exception `x` as a Julia `Exception`.
"""
mutable struct PyException <: Exception
    _t::Py
    _v::Py
    _b::Py
    _isnormalized::Bool
end
