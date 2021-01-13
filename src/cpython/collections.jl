PyABC_Register(s, t) = begin
    r = PyObject_GetAttrString(t, "register")
    isnull(r) && return Cint(-1)
    u = PyObject_CallNice(r, PyObjectRef(s))
    Py_DecRef(r)
    isnull(u) && return Cint(-1)
    Py_DecRef(u)
    Cint(0)
end

for n in [:Container, :Hashable, :Iterable, :Iterator, :Reversible, :Generator, :Sized,
    :Callable, :Collection, :Sequence, :MutableSequence, :ByteString, :Set, :MutableSet,
    :Mapping, :MutableMapping, :MappingView, :ItemsView, :KeysView, :ValuesView,
    :Awaitable, :Coroutine, :AsyncIterable, :AsyncIterator, :AsyncGenerator]
    p = Symbol(:Py, n, :ABC)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyPtr())
    @eval $t(doimport::Bool=true) = begin
        ptr = $tr[]
        isnull(ptr) || return ptr
        a = doimport ? PyImport_ImportModule("collections.abc") : PyImport_GetModule("collections.abc")
        isnull(a) && return a
        b = PyObject_GetAttrString(a, $(string(n)))
        Py_DecRef(a)
        isnull(b) && return b
        $tr[] = b
    end
    @eval $c(o) = begin
        t = $t(false)
        isnull(t) && return (PyErr_IsSet() ? Cint(-1) : Cint(0))
        PyObject_IsInstance(o, t)
    end
end

"""
    PyIterable_Collect(xs::PyPtr, T::Type, [skip=false]) :: Vector{T}

Convert the elements of `xs` to type `T` and collect them into a vector.
On error, an empty vector is returned.

If `skip` then elements which cannot be converted to a `T` are skipped over,
instead of raising an error. Other errors are still propagated.
"""
PyIterable_Collect(xso::PyPtr, ::Type{T}, skip::Bool=false) where {T} = begin
    xs = T[]
    it = PyObject_GetIter(xso)
    isnull(it) && return xs
    try
        while true
            xo = PyIter_Next(it)
            if !isnull(xo)
                if skip
                    r = PyObject_TryConvert(xo, T)
                    Py_DecRef(xo)
                    r == -1 && (empty!(xs); break)
                    r ==  0 && continue
                else
                    r = PyObject_Convert(xo, T)
                    Py_DecRef(xo)
                    r == -1 && (empty!(xs); break)
                end
                x = takeresult(T)
                push!(xs, x)
            else
                PyErr_IsSet() && empty!(xs)
                break
            end
        end
    catch err
        empty!(xs)
        PyErr_SetJuliaError(err)
    finally
        Py_DecRef(it)
    end
    return xs
end

PyIterable_ConvertRule_vector(o, ::Type{S}) where {S<:Vector} = begin
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
            return putresult(xs)
        end
    end
end
PyIterable_ConvertRule_vector(o, ::Type{Vector}) =
    PyIterable_ConvertRule_vector(o, Vector{Python.PyObject})

PyIterable_ConvertRule_set(o, ::Type{S}) where {S<:Set} = begin
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
            return putresult(xs)
        end
    end
end
PyIterable_ConvertRule_set(o, ::Type{Set}) =
    PyIterable_ConvertRule_set(o, T, Set{Python.PyObject})

PyIterable_ConvertRule_tuple(o, ::Type{S}) where {S<:Tuple} = begin
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
            return putresult(S(xs))
        end
    end
end
PyIterable_ConvertRule_tuple(o, ::Type{Tuple{}}) = putresult(())

PyIterable_ConvertRule_pair(o, ::Type{Pair{K,V}}) where {K,V} = begin
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
    putresult(Pair{K,V}(k, v))
end
PyIterable_ConvertRule_pair(o, ::Type{Pair{K}}) where {K} =
    PyIterable_ConvertRule_pair(o, Pair{K,Python.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{Pair{K,V} where K}) where {V} =
    PyIterable_ConvertRule_pair(o, Pair{Python.PyObject,V})
PyIterable_ConvertRule_pair(o, ::Type{Pair}) =
    PyIterable_ConvertRule_pair(o, Pair{Python.PyObject,Python.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{S}) where {S<:Pair} = begin
    PyErr_SetString(PyExc_Exception(), "When converting Python iterable to Julia 'Pair', the destination type cannot be too complicated: the two types must either be fully specified or left unspecified. Got '$S'.")
    return -1
end

PyMapping_ConvertRule_dict(o, ::Type{S}) where {S<:Dict} = begin
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
            return putresult(xs)
        end
    end
end
PyMapping_ConvertRule_dict(o, ::Type{Dict{K}}) where {K} =
    PyMapping_ConvertRule_dict(o, Dict{K,Python.PyObject})
PyMapping_ConvertRule_dict(o, ::Type{Dict{K,V} where K}) where {V} =
    PyMapping_ConvertRule_dict(o, Dict{Python.PyObject,V})
PyMapping_ConvertRule_dict(o, ::Type{Dict}) =
    PyMapping_ConvertRule_dict(o, Dict{Python.PyObject,Python.PyObject})
