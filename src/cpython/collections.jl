for n in [:Container, :Hashable, :Iterable, :Iterator, :Reversible, :Generator, :Sized,
    :Callable, :Collection, :Sequence, :MutableSequence, :ByteString, :Set, :MutableSet,
    :Mapping, :MutableMapping, :MappingView, :ItemsView, :KeysView, :ValuesView,
    :Awaitable, :Coroutine, :AsyncIterable, :AsyncIterator, :AsyncGenerator]
    p = Symbol(:Py, n, :ABC)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyPtr())
    @eval $t() = begin
        ptr = $tr[]
        isnull(ptr) || return ptr
        a = PyImport_ImportModule("collections.abc")
        isnull(a) && return a
        b = PyObject_GetAttrString(a, $(string(n)))
        Py_DecRef(a)
        isnull(b) && return b
        $tr[] = b
    end
    @eval $c(o) = begin
        t = $t()
        isnull(t) && return Cint(-1)
        PyObject_IsInstance(o, t)
    end
end

PyIterable_ConvertRule_vector(o, ::Type{T}, ::Type{S}) where {T, S<:Vector} = begin
    it = PyObject_GetIter(o)
    isnull(it) && return -1
    xs = S()
    while true
        xo = PyIter_Next(it)
        if !isnull(xo)
            r = PyObject_TryConvert(xo, eltype(xs))
            Py_DecRef(xo)
            r == 1 || (Py_DecRef(it); return r)
            x = takeresult(eltype(xs))
            push!(xs, x)
        elseif PyErr_IsSet()
            Py_DecRef(it)
            return -1
        else
            Py_DecRef(it)
            return putresult(T, xs)
        end
    end
end
PyIterable_ConvertRule_vector(o, ::Type{T}, ::Type{Vector}) where {T} =
    PyIterable_ConvertRule_vector(o, T, Vector{Python.PyObject})

PyIterable_ConvertRule_set(o, ::Type{T}, ::Type{S}) where {T, S<:Set} = begin
    it = PyObject_GetIter(o)
    isnull(it) && return -1
    xs = S()
    while true
        xo = PyIter_Next(it)
        if !isnull(xo)
            r = PyObject_TryConvert(xo, eltype(xs))
            Py_DecRef(xo)
            r == 1 || (Py_DecRef(it); return r)
            x = takeresult(eltype(xs))
            push!(xs, x)
        elseif PyErr_IsSet()
            Py_DecRef(it)
            return -1
        else
            Py_DecRef(it)
            return putresult(T, xs)
        end
    end
end
PyIterable_ConvertRule_set(o, ::Type{T}, ::Type{Set}) where {T} =
    PyIterable_ConvertRule_set(o, T, Set{Python.PyObject})

PyIterable_ConvertRule_tuple(o, ::Type{T}, ::Type{S}) where {T,S<:Tuple} = begin
    if !(Tuple isa DataType)
        PyErr_SetString(PyExc_Exception(), "When converting Python 'tuple' to Julia 'Tuple', the destination type must be a 'DataType', i.e. not parametric and not a union. Got '$S'.")
        return -1
    end
    ts = S.parameters
    if !isempty(ts) && Base.isvarargtype(ts[end])
        isvararg = true
        vartype = ts[end].body.parameters[1]
        ts = ts[1:end-1]
    else
        isvararg = false
    end
    it = PyObject_GetIter(o)
    isnull(it) && return -1
    xs = Union{ts..., isvararg ? vartype : Union{}}[]
    i = 0
    while true
        xo = PyIter_Next(it)
        if !isnull(xo)
            i += 1
            if i ≤ length(ts)
                t = ts[i]
            elseif isvararg
                t = vartype
            else
                Py_DecRef(it)
                Py_DecRef(xo)
                return -1
            end
            r = PyObject_TryConvert(xo, t)
            Py_DecRef(xo)
            r == 1 || (Py_DecRef(it); return r)
            x = takeresult(t)
            push!(xs, x)
        elseif PyErr_IsSet()
            Py_DecRef(it)
            return -1
        else
            Py_DecRef(it)
            return putresult(T, S(xs))
        end
    end
end
PyIterable_ConvertRule_tuple(o, ::Type{T}, ::Type{Tuple{}}) where {T} = putresult(T, ())

PyIterable_ConvertRule_pair(o, ::Type{T}, ::Type{Pair{K,V}}) where {T,K,V} = begin
    it = PyObject_GetIter(o)
    isnull(it) && return -1
    # get the first item
    ko = PyIter_Next(it)
    if isnull(ko)
        Py_DecRef(it)
        return PyErr_IsSet() ? -1 : 0
    end
    # convert it
    r = PyObject_TryConvert(ko, K)
    Py_DecRef(ko)
    if r != 1
        Py_DecRef(it)
        return r
    end
    k = takeresult(K)
    # get the second item
    vo = PyIter_Next(it)
    if isnull(vo)
        Py_DecRef(it)
        return PyErr_IsSet() ? -1 : 0
    end
    # convert it
    r = PyObject_TryConvert(vo, V)
    Py_DecRef(vo)
    if r != 1
        Py_DecRef(it)
        return r
    end
    v = takeresult(V)
    # too many values?
    xo = PyIter_Next(it)
    if !isnull(xo)
        Py_DecRef(xo)
        Py_DecRef(it)
        return 0
    end
    # done
    Py_DecRef(it)
    putresult(T, Pair{K,V}(k, v))
end
PyIterable_ConvertRule_pair(o, ::Type{T}, ::Type{Pair{K}}) where {T,K} =
    PyIterable_ConvertRule_pair(o, T, Pair{K,Python.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{T}, ::Type{Pair{K,V} where K}) where {T,V} =
    PyIterable_ConvertRule_pair(o, T, Pair{Python.PyObject,V})
PyIterable_ConvertRule_pair(o, ::Type{T}, ::Type{Pair}) where {T} =
    PyIterable_ConvertRule_pair(o, T, Pair{Python.PyObject,Python.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{T}, ::Type{S}) where {T,S<:Pair} = begin
    PyErr_SetString(PyExc_Exception(), "When converting Python iterable to Julia 'Pair', the destination type cannot be too complicated: the two types must either be fully specified or left unspecified. Got '$S'.")
    return -1
end

PyMapping_ConvertRule_dict(o, ::Type{T}, ::Type{S}) where {T, S<:Dict} = begin
    it = PyObject_GetIter(o)
    isnull(it) && return -1
    xs = S()
    while true
        ko = PyIter_Next(it)
        if !isnull(ko)
            # get the key
            r = PyObject_TryConvert(ko, keytype(xs))
            r == 1 || (Py_DecRef(it); Py_DecRef(ko); return r)
            k = takeresult(keytype(xs))
            # get the value
            vo = PyObject_GetItem(o, ko)
            isnull(vo) && (Py_DecRef(it); Py_DecRef(ko); return -1)
            r = PyObject_TryConvert(vo, valtype(xs))
            Py_DecRef(vo)
            Py_DecRef(ko)
            r == 1 || (Py_DecRef(it); return -1)
            v = takeresult(valtype(xs))
            xs[k] = v
        elseif PyErr_IsSet()
            Py_DecRef(it)
            return -1
        else
            Py_DecRef(it)
            return putresult(T, xs)
        end
    end
end
PyMapping_ConvertRule_dict(o, ::Type{T}, ::Type{Dict{K}}) where {T,K} =
    PyMapping_ConvertRule_dict(o, T, Dict{K,Python.PyObject})
PyMapping_ConvertRule_dict(o, ::Type{T}, ::Type{Dict{K,V} where K}) where {T,V} =
    PyMapping_ConvertRule_dict(o, T, Dict{Python.PyObject,V})
PyMapping_ConvertRule_dict(o, ::Type{T}, ::Type{Dict}) where {T} =
    PyMapping_ConvertRule_dict(o, T, Dict{Python.PyObject,Python.PyObject})
