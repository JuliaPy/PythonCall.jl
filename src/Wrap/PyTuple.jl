PyTuple(x = pytuple()) = PyTuple{Tuple}(x)

ispy(::PyTuple) = true
Py(x::PyTuple) = x.py

@generated function static_length(::PyTuple{T}) where {T}
    try
        fieldcount(T)
    catch
        nothing
    end
end

@generated function min_length(::PyTuple{T}) where {T}
    count(!Base.isvarargtype, T.parameters)
end

function check_length(x::PyTuple)
    len = pylen(x.py)
    explen = PythonCall.Wrap.static_length(x)
    if explen === nothing
        minlen = PythonCall.Wrap.min_length(x)
        len ≥ minlen
    else
        len == explen
    end
end

Base.IteratorSize(::Type{<:PyTuple}) = Base.HasLength()

Base.length(x::PyTuple{T}) where {T<:Tuple} =
    @something(static_length(x), max(min_length(x), Int(pylen(x.py))))

Base.IteratorEltype(::Type{<:PyTuple}) = Base.HasEltype()

Base.eltype(::Type{PyTuple{T}}) where {T<:Tuple} = eltype(T)

Base.checkbounds(::Type{Bool}, x::PyTuple, i::Integer) = 1 ≤ i ≤ length(x)

Base.checkbounds(x::PyTuple, i::Integer) =
    if !checkbounds(Bool, x, i)
        throw(BoundsError(x, i))
    end

Base.@propagate_inbounds function Base.getindex(x::PyTuple{T}, i::Integer) where {T<:Tuple}
    i = convert(Int, i)::Int
    @boundscheck checkbounds(x, i)
    E = fieldtype(T, i)
    return pyconvert(E, @py x[@jl(i - 1)])
end

Base.@propagate_inbounds function Base.setindex!(
    x::PyTuple{T},
    v,
    i::Integer,
) where {T<:Tuple}
    i = convert(Int, i)::Int
    @boundscheck checkbounds(x, i)
    E = fieldtype(T, i)
    v = convert(E, v)::E
    @py x[@jl(i - 1)] = v
    x
end

function Base.iterate(x::PyTuple{T}, ni = (length(x), 1)) where {T<:Tuple}
    n, i = ni
    if i > @something(static_length(x), n)
        nothing
    else
        (x[i], (n, i + 1))
    end
end

function Base.Tuple(x::PyTuple{T}) where {T<:Tuple}
    n = static_length(x)
    if n === nothing
        ntuple(i -> x[i], length(x))::T
    else
        ntuple(i -> x[i], Val(n))::T
    end
end

# Conversion rule for builtins:tuple -> PyTuple
function pyconvert_rule_tuple(
    ::Type{T},
    x::Py,
    ::Type{T1} = Utils._type_ub(T),
) where {T<:PyTuple,T1}
    ans = @inbounds T1(x)
    if check_length(ans)
        pyconvert_return(ans)
    else
        pyconvert_unconverted()
    end
end

function Base.show(io::IO, mime::MIME"text/plain", x::PyTuple)
    if !(get(io, :typeinfo, Any) <: PyTuple)
        print(io, "PyTuple: ")
    end
    show(io, mime, Tuple(x))
    nothing
end
