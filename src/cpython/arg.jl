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
        it = PyObject_GetIter(kwargs)
        isnull(it) && return -1
        argnames = String[]
        while true
            argnameo = PyIter_Next(it)
            if !isnull(argnameo)
                argname = PyUnicode_AsString(argnameo)
                Py_DecRef(argnameo)
                isempty(argname) && PyErr_IsSet() && (Py_DecRef(it); return -1)
                push!(argnames, argname)
            elseif PyErr_IsSet()
                Py_DecRef(it)
                return -1
            else
                Py_DecRef(it)
                break
            end
        end
        PyErr_SetString(PyExc_TypeError(), "$name() got unexpected keyword arguments: $(join(["'$n'" for n in argnames], ", "))")
        return -1
    end
end

struct NODEFAULT end

PyArg_Find(args::PyPtr, kwargs::PyPtr, i::Union{Int,Nothing}, k::Union{String,Nothing}) =
    if i !== nothing && !isnull(args) && 0 ≤ i ≤ PyTuple_Size(args)
        return PyTuple_GetItem(args, i)
    elseif k !== nothing && !isnull(kwargs) && (ro = PyDict_GetItemString(kwargs, k)) != PyPtr()
        return ro
    else
        return PyPtr()
    end

PyArg_GetArg(::Type{T}, name::String, args::PyPtr, kwargs::PyPtr=PyPtr(), i::Union{Int,Nothing}=nothing, k::Union{String,Nothing}=nothing, d::Union{T,NODEFAULT}=NODEFAULT()) where {T} = begin
    ro = PyArg_Find(args, kwargs, i, k)
    if isnull(ro)
        if k !== nothing
            PyErr_SetString(PyExc_TypeError(), "$name() did not get required argument '$k'")
        elseif i !== nothing
            PyErr_SetString(PyExc_TypeError(), "$name() takes at least $(i+1) arguments (got $(isnull(args) ? 0 : PyTuple_Size(args)))")
        else
            error("impossible to satisfy this argument")
        end
        return -1
    end
    r = PyObject_TryConvert(ro, T)
    if r == -1
        return -1
    elseif r == 0
        if d === NODEFAULT()
            PyErr_SetString(PyExc_TypeError(), "Argument $(k !== nothing ? "'$k'" : i !== nothing ? "$i" : error("impossible")) to $name() must be convertible to a Julia '$T'")
            return -1
        else
            putresult(d)
            return 0
        end
    else
        return 0
    end
end
PyArg_GetArg(::Type{T}, name::String, args::PyPtr, i::Union{Int,Nothing}, k::Union{String,Nothing}=nothing, d::Union{T,NODEFAULT}=NODEFAULT()) where {T} =
    PyArg_GetArg(T, name, args, PyPtr(), i, k, d)
PyArg_GetArg(::Type{T}, name::String, args::PyPtr, kwargs::PyPtr, k::Union{String,Nothing}, d::Union{T, NODEFAULT}=NODEFAULT()) where {T} =
    PyArg_GetArg(T, name, args, kwargs, nothing, k, d)
