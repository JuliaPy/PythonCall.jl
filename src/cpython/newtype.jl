### CACHE

cacheptr!(c, x::Ptr) = x
cacheptr!(c, x::String) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractString) = cacheptr!(c, String(x))
cacheptr!(c, x::Array) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractArray) = cacheptr!(c, Array(x))
cacheptr!(c, x::PyObject) = (push!(c, x); pyptr(x))
cacheptr!(c, x::Base.CFunction) = (push!(c, x); Base.unsafe_convert(Ptr{Cvoid}, x))

macro cfunctionOO(f)
    :(@cfunction($f, PyPtr, (PyPtr,)))
end

macro cfunctionOOO(f)
    :(@cfunction($f, PyPtr, (PyPtr, PyPtr)))
end

macro cfunctionOOP(f)
    :(@cfunction($f, PyPtr, (PyPtr, Ptr{Cvoid})))
end

macro cfunctionOOOO(f)
    :(@cfunction($f, PyPtr, (PyPtr, PyPtr, PyPtr)))
end

macro cfunctionOOOI(f)
    :(@cfunction($f, PyPtr, (PyPtr, PyPtr, Cint)))
end

macro cfunctionIO(f)
    :(@cfunction($f, Cint, (PyPtr,)))
end

macro cfunctionIOO(f)
    :(@cfunction($f, Cint, (PyPtr, PyPtr)))
end

macro cfunctionIOOO(f)
    :(@cfunction($f, Cint, (PyPtr, PyPtr, PyPtr)))
end

macro cfunctionVO(f)
    :(@cfunction($f, Cvoid, (PyPtr,)))
end

macro cfunctionZO(f)
    :(@cfunction($f, Py_ssize_t, (PyPtr,)))
end

"""
    LazyPyObject(f)

A Python object which is constructed lazily.

The first time it is accessed (by calling it), it is generated as `f()::PyPtr`.

In particular, we use this for defining new types. These types contain `@cfunction`s,
and sometimes there is a path from these functions back to the fuction to produce the type,
which calls `@cfunction`. The Julia compiler really doesn't like this, sometimes getting
into an infinte loop, or corrupting memory some other way. By putting the generation
function in an `::Any` variable, the function is hidden from type inference and all is fine.
"""
mutable struct LazyPyObject
    ptr :: PyPtr
    make :: Any
    LazyPyObject(f) = begin
        precompile(f, ())
        new(PyNULL, f)
    end
end

(t::LazyPyObject)() = begin
    if isnull(t.ptr)
        t.ptr = t.make()::PyPtr
    end
    return t.ptr
end
