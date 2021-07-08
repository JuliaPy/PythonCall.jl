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

_PyIterable_ConvertRule_vecorset(o, xs, ::Type{T}) where {T} = begin
    @pydsl_nojlerror for xo in (@py externbx o)
        if canconvert(T, xo)
            x = takeresult(T)
            xs = push!!(xs, x)
        else
            return 0
        end
    end onpyerror=(return -1)
    # sometimes push!! can make the eltype larger than desired
    # TODO: we can have S=Vector{<:Union{Integer,Vector{<:Integer}}} and T2=S,
    #   can we make it be T2=Vector{Union{Int,Vector{Int}}} instead (i.e. only grow the types inside the Union)
    # TODO: roll our own push!! that does this, instead of shrinking the eltype post-hoc
    if !(eltype(xs) <: T)
        T2 = _typeintersect(eltype(xs), T)
        xs = xs isa Set ? Set{T2}(xs) : Vector{T2}(xs)
    end
    putresult(xs)
end
PyIterable_ConvertRule_vecorset(o, ::Type{S}) where {S} = _PyIterable_ConvertRule_vecorset(o, _type_lb(S)(), eltype(_type_ub(S)))

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
    xs = Any[]
    @pydsl_nojlerror for xo in (@py externbx o)
        if length(xs) < length(ts)
            t = ts[length(xs) + 1]
        elseif isvararg
            t = vartype
        else
            return 0
        end
        if canconvert(t, xo)
            x = takeresult(t)
            push!(xs, x)
        else
            return 0
        end
    end onpyerror=(return -1)
    length(xs) < length(ts) ? 0 : putresult(S(xs))
end

PyIterable_ConvertRule_namedtuple(o, ::Type{NamedTuple{names, types}}) where {names, types<:Tuple} = begin
    r = PyIterable_ConvertRule_tuple(o, types)
    r == 1 ? putresult(NamedTuple{names, types}(takeresult(types))) : r
end
PyIterable_ConvertRule_namedtuple(o, ::Type{NamedTuple{names}}) where {names} = begin
    types = NTuple{length(names), Any}
    r = PyIterable_ConvertRule_tuple(o, types)
    r == 1 ? putresult(NamedTuple{names}(takeresult(types))) : r
end
PyIterable_ConvertRule_namedtuple(o, ::Type{S}) where {S<:NamedTuple} = 0

_PyIterable_ConvertRule_pair(o, ::Type{Pair{K1,V1}}, ::Type{Pair{K2,V2}}) where {K1,V1,K2,V2} = begin
    k = nothing
    v = nothing
    i = 0
    @pydsl_nojlerror for xo in (@py externbx o)
        i += 1
        if i == 1
            # key
            if canconvert(K2, xo)
                k = takeresult(K2)
            else
                return 0
            end
        elseif i == 2
            # value
            if canconvert(V2, xo)
                v = takeresult(V2)
            else
                return 0
            end
        else
            # too many items
            return 0
        end
    end onpyerror=(return -1)
    if i == 2
        k::K2
        v::V2
        K = K1==K2 ? K1 : K1==Union{} ? typeof(k) : typeintersect(K2, promote_type(K1, typeof(k)))
        V = V1==V2 ? V1 : V1==Union{} ? typeof(v) : typeintersect(V2, promote_type(V1, typeof(v)))
        putresult(Pair{K,V}(k, v))
    else
        return 0
    end
end
PyIterable_ConvertRule_pair(o, ::Type{S}) where {S<:Pair} = _PyIterable_ConvertRule_pair(o, _type_lb(S), _type_ub(S))

_PyMapping_ConvertRule_dict(o, xs, ::Type{K}, ::Type{V}) where {K,V} = begin
    @pydsl_nojlerror for ko in (@py externbx o)
        # key
        if canconvert(K, ko)
            k = takeresult(K)
        else
            return 0
        end
        # value
        if canconvert(V, (@py externbx o)[ko])
            v = takeresult(V)
        else
            return 0
        end
        xs = push!!(xs, k => v)
    end
    # sometimes push!! can make the keytype/valtype larger than desired
    # TODO: same todo as in vecorset above
    if !(keytype(xs) <: K && valtype(xs) <: V)
        K2 = _typeintersect(keytype(xs), K)
        V2 = _typeintersect(valtype(xs), V)
        xs = Dict{K2,V2}(xs)
    end
    putresult(xs)
end
PyMapping_ConvertRule_dict(o, ::Type{S}) where {S} = _PyMapping_ConvertRule_dict(o, _type_lb(S)(), keytype(_type_ub(S)), valtype(_type_ub(S)))
