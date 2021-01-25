PyArg_CheckNumArgsEq(name::String, args::PyPtr, n::Integer) = begin
    m = isnull(args) ? 0 : PyTuple_Size(args)
    if m == n
        return 0
    else
        PyErr_SetString(PyExc_TypeError(), "$name() takes $n arguments ($m given)")
        return -1
    end
end

PyArg_CheckNumArgsGe(name::String, args::PyPtr, n::Integer) = begin
    m = isnull(args) ? 0 : PyTuple_Size(args)
    if m ≥ n
        return 0
    else
        PyErr_SetString(PyExc_TypeError(), "$name() takes at least $n arguments ($m given)")
        return -1
    end
end

PyArg_CheckNumArgsLe(name::String, args::PyPtr, n::Integer) = begin
    m = isnull(args) ? 0 : PyTuple_Size(args)
    if m ≤ n
        return 0
    else
        PyErr_SetString(PyExc_TypeError(), "$name() takes at most $n arguments ($m given)")
        return -1
    end
end

PyArg_CheckNumArgsBetween(name::String, args::PyPtr, n0::Integer, n1::Integer) = begin
    m = isnull(args) ? 0 : PyTuple_Size(args)
    if n0 ≤ m ≤ n1
        return 0
    else
        PyErr_SetString(PyExc_TypeError(), "$name() takes $n0 to $n1 arguments ($m given)")
        return -1
    end
end

PyArg_CheckNoKwargs(name::String, kwargs::PyPtr) = begin
    if isnull(kwargs) || PyObject_Length(kwargs) == 0
        return 0
    else
        argnames = PyIterable_Collect(kwargs, String)
        isempty(argnames) && PyErr_IsSet() && return -1
        PyErr_SetString(
            PyExc_TypeError(),
            "$name() got unexpected keyword arguments: $(join(["'$n'" for n in argnames], ", "))",
        )
        return -1
    end
end

struct NODEFAULT end

PyArg_Find(args::PyPtr, kwargs::PyPtr, i::Union{Int,Nothing}, k::Union{String,Nothing}) =
    if i !== nothing && !isnull(args) && 0 ≤ i < PyTuple_Size(args)
        return PyTuple_GetItem(args, i)
    elseif k !== nothing &&
           !isnull(kwargs) &&
           (ro = PyDict_GetItemString(kwargs, k)) != PyPtr()
        return ro
    else
        return PyPtr()
    end

"""
    PyArg_GetArg(T, funcname, [args, i], [kwargs, k], [default])

Attempt to find and convert the specified argument to a `T`. Return 0 on success, in which case the result can be retrieved with `takeresult(T)`. Return -1 on failure, with a Python error set.

- `funcname::String` is the name of the function this is used in, for constructing error messages.
- `args::PyPtr` is a tuple of arguments, and `i::Int` an index.
- `kwargs::PyPtr` is a dict of keyword arguments, and `k::String` a key.
- `default` specifies a default value if the argument is not found. NOTE: It need not be a `T`, so when retrieving the result use `takeresult(Union{T,typeof(default)})`.
"""
PyArg_GetArg(
    ::Type{T},
    name::String,
    args::PyPtr,
    i::Union{Int,Nothing},
    kwargs::PyPtr,
    k::Union{String,Nothing},
    d = NODEFAULT(),
) where {T} = begin
    ro = PyArg_Find(args, kwargs, i, k)
    if isnull(ro)
        if d !== NODEFAULT()
            putresult(d)
            return 0
        elseif k !== nothing
            PyErr_SetString(PyExc_TypeError(), "$name() did not get required argument '$k'")
            return -1
        elseif i !== nothing && i ≥ 0
            PyErr_SetString(
                PyExc_TypeError(),
                "$name() takes at least $(i+1) arguments (got $(isnull(args) ? 0 : PyTuple_Size(args)))",
            )
            return -1
        else
            error("impossible to satisfy this argument")
        end
    end
    r = PyObject_TryConvert(ro, T)
    if r == -1
        return -1
    elseif r == 0
        PyErr_SetString(
            PyExc_TypeError(),
            "Argument $(k !== nothing ? "'$k'" : i !== nothing ? "$i" : error("impossible")) to $name() must be convertible to a Julia '$T'",
        )
        return -1
    else
        return 0
    end
end
PyArg_GetArg(::Type{T}, name::String, args::PyPtr, i::Int, d = NODEFAULT()) where {T} =
    PyArg_GetArg(T, name, args, i, PyPtr(), nothing, d)
PyArg_GetArg(::Type{T}, name::String, kwargs::PyPtr, k::String, d = NODEFAULT()) where {T} =
    PyArg_GetArg(T, name, PyPtr(), nothing, kwargs, k, d)
