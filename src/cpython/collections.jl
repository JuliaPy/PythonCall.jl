PyABC_Register(s, t) = begin
    r = PyObject_GetAttrString(t, "register")
    isnull(r) && return Cint(-1)
    u = PyObject_CallNice(r, PyObjectRef(s))
    Py_DecRef(r)
    isnull(u) && return Cint(-1)
    Py_DecRef(u)
    Cint(0)
end

for n in [
    :Container,
    :Hashable,
    :Iterable,
    :Iterator,
    :Reversible,
    :Generator,
    :Sized,
    :Callable,
    :Collection,
    :Sequence,
    :MutableSequence,
    :ByteString,
    :Set,
    :MutableSet,
    :Mapping,
    :MutableMapping,
    :MappingView,
    :ItemsView,
    :KeysView,
    :ValuesView,
    :Awaitable,
    :Coroutine,
    :AsyncIterable,
    :AsyncIterator,
    :AsyncGenerator,
]
    p = Symbol(:Py, n, :ABC)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyNULL)
    @eval $t(doimport::Bool = true) = begin
        ptr = $tr[]
        isnull(ptr) || return ptr
        a =
            doimport ? PyImport_ImportModule("collections.abc") :
            PyImport_GetModule("collections.abc")
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
    PyIterable_Map(f, xs::PyPtr) :: Int

Calls `f(x)` for each element `x::PyPtr` of `xs`.

The function `f` must return an integer, where -1 signals a Python error,
0 signals to stop iterating, 1 to continue.

If `f` throws a Julia error, it is caught and converted to a Python error.
In this situation, `x` is automatically decref'd, but it is up to the user
to ensure that other references created by `f` are cleared.

Return -1 on error, 0 if iteration was stopped, 1 if it reached the end.
"""
PyIterable_Map(f, xso::PyPtr) = begin
    it = PyObject_GetIter(xso)
    isnull(it) && return PyNULL
    xo = PyNULL
    try
        while true
            xo = PyIter_Next(it)
            if !isnull(xo)
                r = f(xo)
                Py_DecRef(xo)
                xo = PyNULL
                if r == -1
                    return -1
                elseif r == 0
                    return 0
                end
            elseif PyErr_IsSet()
                return -1
            else
                return 1
            end
        end
    catch err
        PyErr_SetJuliaError(err)
        -1
    finally
        Py_DecRef(it)
        Py_DecRef(xo)
    end
end

"""
    PyIterable_Collect(xs::PyPtr, xs, [skip=false]) :: Vector{T}

Convert the elements of `xs` to type `T` and collect them into a vector.
On error, an empty vector is returned.

If `skip` then elements which cannot be converted to a `T` are skipped over,
instead of raising an error. Other errors are still propagated.
"""
PyIterable_Collect(xso::PyPtr, ::Type{T}, skip::Bool = false) where {T} = begin
    xs = T[]
    r = if skip
        PyIterable_Map(xso) do xo
            r = PyObject_TryConvert(xo, eltype(xs))
            r == -1 && return -1
            r == 0 && return 1
            x = takeresult(eltype(xs))
            push!(xs, x)
            return 1
        end
    else
        PyIterable_Map(xso) do xo
            r = PyObject_Convert(xo, eltype(xs))
            r == -1 && return -1
            x = takeresult(eltype(xs))
            push!(xs, x)
            return 1
        end
    end
    r == -1 && empty!(xs)
    xs
end

PyIterable_ConvertRule_vecorset(o, ::Type{S}) where {S<:Union{Vector,Set}} = begin
    xs = S()
    r = PyIterable_Map(o) do xo
        r = PyObject_TryConvert(xo, eltype(xs))
        r == -1 && return -1
        r == 0 && return 0
        x = takeresult(eltype(xs))
        push!(xs, x)
        return 1
    end
    r == 1 && putresult(xs)
    r
end
PyIterable_ConvertRule_vecorset(o, ::Type{Vector}) =
    PyIterable_ConvertRule_vecorset(o, Vector{PythonCall.PyObject})
PyIterable_ConvertRule_vecorset(o, ::Type{Set}) =
    PyIterable_ConvertRule_vecorset(o, Set{PythonCall.PyObject})

PyIterable_ConvertRule_tuple(o, ::Type{S}) where {S<:Tuple} = begin
    S isa DataType || return 0
    ts = S.parameters
    if !isempty(ts) && Base.isvarargtype(ts[end])
        isvararg = true
        vartype = ts[end].body.parameters[1]
        ts = ts[1:end-1]
    else
        isvararg = false
    end
    xs = Union{ts...,isvararg ? vartype : Union{}}[]
    r = PyIterable_Map(o) do xo
        if length(xs) < length(ts)
            t = ts[length(xs)+1]
        elseif isvararg
            t = vartype
        else
            return 0
        end
        r = PyObject_TryConvert(xo, t)
        r == -1 && return -1
        r == 0 && return 0
        x = takeresult(t)
        push!(xs, x)
        return 1
    end
    r == -1 ? -1 : r == 0 ? 0 : length(xs) ≥ length(ts) ? putresult(S(xs)) : 0
end

PyIterable_ConvertRule_namedtuple(o, ::Type{NamedTuple{names, types}}) where {names, types<:Tuple} = begin
    r = PyIterable_ConvertRule_tuple(o, types)
    r == 1 ? putresult(NamedTuple{names, types}(takeresult(types))) : r
end
PyIterable_ConvertRule_namedtuple(o, ::Type{NamedTuple{names}}) where {names} = begin
    types = NTuple{length(names), PythonCall.PyObject}
    r = PyIterable_ConvertRule_tuple(o, types)
    r == 1 ? putresult(NamedTuple{names}(takeresult(types))) : r
end
PyIterable_ConvertRule_namedtuple(o, ::Type{S}) where {S<:NamedTuple} = 0

PyIterable_ConvertRule_pair(o, ::Type{Pair{K,V}}) where {K,V} = begin
    k = Ref{K}()
    v = Ref{V}()
    i = Ref(0)
    r = PyIterable_Map(o) do xo
        if i[] == 0
            # key
            r = PyObject_TryConvert(xo, eltype(k))
            r == -1 && return -1
            r == 0 && return 0
            i[] = 1
            k[] = takeresult(eltype(k))
            return 1
        elseif i[] == 1
            # value
            r == PyObject_TryConvert(xo, eltype(v))
            r == -1 && return -1
            r == 0 && return 0
            i[] = 2
            v[] = takeresult(eltype(v))
            return 1
        else
            return 0
        end
    end
    r == -1 ? -1 : r == 0 ? 0 : i[] == 2 ? putresult(Pair{K,V}(k[], v[])) : 0
end
PyIterable_ConvertRule_pair(o, ::Type{Pair{K}}) where {K} =
    PyIterable_ConvertRule_pair(o, Pair{K,PythonCall.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{Pair{K,V} where K}) where {V} =
    PyIterable_ConvertRule_pair(o, Pair{PythonCall.PyObject,V})
PyIterable_ConvertRule_pair(o, ::Type{Pair}) =
    PyIterable_ConvertRule_pair(o, Pair{PythonCall.PyObject,PythonCall.PyObject})
PyIterable_ConvertRule_pair(o, ::Type{S}) where {S<:Pair} = begin
    PyErr_SetString(
        PyExc_Exception(),
        "When converting Python iterable to Julia 'Pair', the destination type cannot be too complicated: the two types must either be fully specified or left unspecified. Got '$S'.",
    )
    return -1
end

PyMapping_ConvertRule_dict(o, ::Type{S}) where {S<:Dict} = begin
    xs = S()
    r = PyIterable_Map(o) do ko
        # get the key
        r = PyObject_TryConvert(ko, keytype(xs))
        r == -1 && return -1
        r == 0 && return 0
        k = takeresult(keytype(xs))
        # get the value
        vo = PyObject_GetItem(o, ko)
        isnull(vo) && return -1
        r = PyObject_TryConvert(vo, valtype(xs))
        Py_DecRef(vo)
        r == -1 && return -1
        r == 0 && return 0
        v = takeresult(valtype(xs))
        # done
        xs[k] = v
    end
    r == -1 ? -1 : r == 0 ? 0 : putresult(xs)
end
PyMapping_ConvertRule_dict(o, ::Type{Dict{K}}) where {K} =
    PyMapping_ConvertRule_dict(o, Dict{K,PythonCall.PyObject})
PyMapping_ConvertRule_dict(o, ::Type{Dict{K,V} where K}) where {V} =
    PyMapping_ConvertRule_dict(o, Dict{PythonCall.PyObject,V})
PyMapping_ConvertRule_dict(o, ::Type{Dict}) =
    PyMapping_ConvertRule_dict(o, Dict{PythonCall.PyObject,PythonCall.PyObject})
