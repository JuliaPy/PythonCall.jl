for p in [:Container, :Hashable, :Iterable, :Iterator, :Reversible, :Generator, :Sized, :Callable, :Collection, :Sequence, :MutableSequence, :ByteString, :Set, :MutableSet, :Mapping, :MutableMapping, :MappingView, :ItemsView, :KeysView, :ValuesView, :Awaitable, :Coroutine, :AsyncIterable, :AsyncIterator, :AsyncGenerator]
    j = Symbol(:py, lowercase(string(p)), :abc)
    @eval const $j = PyLazyObject(() -> pycollectionsabcmodule.$p)
    @eval export $j
end

function pyiterable_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    # check subclasses
    if pyisinstance(o, pymappingabc)
        r = pymapping_tryconvert(T, o)
        r === PyConvertFail() || return r
    elseif pyisinstance(o, pysequenceabc)
        r = pysequence_tryconvert(T, o)
        r === PyConvertFail() || return r
    elseif pyisinstance(o, pysetabc)
        r = pyabstractset_tryconvert(T, o)
        r === PyConvertFail() || return r
    end
    # generic conversions
    if (S = _typeintersect(T, PyIterable)) != Union{}
        return S(o)
    elseif (S = _typeintersect(T, Vector{PyObject})) != Union{}
        r = S()
        append!(r, PyIterable{PyObject}(o))
    elseif (S = _typeintersect(T, Vector)) != Union{}
        r = S()
        append!(r, PyIterable{eltype(r)}(o))
    elseif (S = _typeintersect(T, Set{PyObject})) != Union{}
        r = S()
        append!(r, PyIterable{PyObject}(o))
    elseif (S = _typeintersect(T, Set)) != Union{}
        r = S()
        append!(r, PyIterable{eltype(r)}(o))
    elseif (S = _typeintersect(T, Tuple)) != Union{}
        pyiterable_tryconvert(S, o)
    elseif (S = _typeintersect(T, CartesianIndex)) != Union{}
        pyiterable_tryconvert(S, o)
    elseif (S = _typeintersect(T, NamedTuple)) != Union{}
        pyiterable_tryconvert(S, o)
    else
        tryconvert(T, PyIterable(o))
    end
end

# TODO: make parts of this generated for type stability (i.e. the parts dealing with type logic)
function pyiterable_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:Tuple}
    # union?
    if T isa Union
        a = pyiterable_tryconvert(T.a, o)
        b = pyiterable_tryconvert(T.b, o)
        if typeof(a) == typeof(b)
            return a
        elseif a === PyConvertFail()
            return b
        elseif b === PyConvertFail()
            return a
        else
            error("ambiguous conversion")
        end
    end
    # flatten out any type vars
    S = _type_flatten_tuple(T) :: DataType
    # determine component types
    if length(S.parameters) > 0 && Base.isvarargtype(S.parameters[end])
        nfixed = length(S.parameters) - 1
        vartype = S.parameters[end].body.parameters[1]
    else
        nfixed = length(S.parameters)
        vartype = nothing
    end
    # convert components
    xs = []
    for (i,xo) in enumerate(o)
        if i ≤ nfixed
            x = pytryconvert(S.parameters[i], xo)
            x === PyConvertFail() ? (return x) : push!(xs, x)
        elseif vartype === nothing
            # too many values
            return PyConvertFail()
        else
            x = pytryconvert(vartype, xo)
            x === PyConvertFail() ? (return x) : push!(xs, x)
        end
    end
    # check we got enough
    length(xs) ≥ nfixed || return PyConvertFail()
    # success!
    tryconvert(T, Tuple(xs))
end

function pyiterable_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:CartesianIndex}
    x = pyiterable_tryconvert(Tuple{Vararg{Int}}, o)
    x === PyConvertFail() ? x : convert(T, CartesianIndex(x))
end

function pyiterable_tryconvert(::Type{CartesianIndex{N}}, o::AbstractPyObject) where {N}
    x = pyiterable_tryconvert(NTuple{N,Int}, o)
    x === PyConvertFail() ? x : CartesianIndex{N}(x)
end

function pyiterable_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:NamedTuple}
    PyConvertFail()
end

function pyiterable_tryconvert(::Type{T}, o::AbstractPyObject) where {names, T<:NamedTuple{names}}
    x = pyiterable_tryconvert(NTuple{length(names), PyObject}, o)
    x === PyConvertFail() ? x : convert(T, NamedTuple{names}(x))
end

function pyiterable_tryconvert(::Type{NamedTuple{names,Ts}}, o::AbstractPyObject) where {names, Ts<:Tuple}
    x = pyiterable_tryconvert(Ts, o)
    x === PyConvertFail() ? x : NamedTuple{names,Ts}(x)
end

function pymapping_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if (S = _typeintersect(T, AbstractDict)) != Union{}
        pymapping_tryconvert(S, o)
    else
        PyConvertFail()
    end
end

function pymapping_tryconvert(::Type{T}, o::AbstractPyObject) where {T<:AbstractDict}
    tryconvert(T, PyDict(o))
end

function pymapping_tryconvert(::Type{T}, o::AbstractPyObject) where {K, T<:AbstractDict{K}}
    # often fails because e.g. convert(::Dict{String}, ::PyDict{String,Int}) fails
    tryconvert(T, PyDict{K}(o))
end

function pymapping_tryconvert(::Type{T}, o::AbstractPyObject) where {K, V, T<:AbstractDict{K,V}}
    tryconvert(T, PyDict{K,V}(o))
end

function pysequence_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if (S = _typeintersect(T, PyList)) != Union{}
        S(o)
    else
        PyConvertFail()
    end
end

function pyabstractset_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    PyConvertFail()
end
