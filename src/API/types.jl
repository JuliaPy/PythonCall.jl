# Convert

@enum PyConvertPriority begin
    PYCONVERT_PRIORITY_WRAP = 400
    PYCONVERT_PRIORITY_ARRAY = 300
    PYCONVERT_PRIORITY_CANONICAL = 200
    PYCONVERT_PRIORITY_NORMAL = 0
    PYCONVERT_PRIORITY_FALLBACK = -100
end

# Core

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
    Py(::Val{:new}, ptr::Ptr{Cvoid}) = finalizer(Core.py_finalizer, new(ptr))
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

"""
    pybuiltins

An object whose fields are the Python builtins, of type [`Py`](@ref).

For example `pybuiltins.None`, `pybuiltins.int`, `pybuiltins.ValueError`.
"""
baremodule pybuiltins
end
