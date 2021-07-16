# Compatability for Julia below 1.3.
if VERSION < v"1.3"
    allocatedinline(::Type{T}) where {T} = (Base.@_pure_meta; ccall(:jl_array_store_unboxed, Cint, (Any,), T) != Cint(0))
    function aligned_sizeof(T)
        Base.@_pure_meta
        if Base.isbitsunion(T)
            sz = Ref{Csize_t}(0)
            algn = Ref{Csize_t}(0)
            ccall(:jl_islayout_inline, Cint, (Any, Ptr{Csize_t}, Ptr{Csize_t}), T, sz, algn)
            al = algn[]
            return (sz[] + al - 1) & -al
        elseif allocatedinline(T)
            al = Base.datatype_alignment(T)
            return (Core.sizeof(T) + al - 1) & -al
        else
            return Core.sizeof(Ptr{Cvoid})
        end
    end
else
    const allocatedinline = Base.allocatedinline
    const aligned_sizeof = Base.aligned_sizeof
end




### TYPE UTILITIES

# Used to signal a Python error from functions that return general Julia objects
struct PYERR end

# Used to signal that conversion failed
struct NOTIMPLEMENTED end

# Somewhere to stash results
const RESULT = Ref{Any}(nothing)
putresult(x) = (RESULT[] = x; 1)
putresult(x::PYERR) = -1
putresult(x::NOTIMPLEMENTED) = 0
takeresult(::Type{T} = Any) where {T} = (r = RESULT[]::T; RESULT[] = nothing; r)

tryconvert(::Type{T}, x::PYERR) where {T} = PYERR()
tryconvert(::Type{T}, x::NOTIMPLEMENTED) where {T} = NOTIMPLEMENTED()
tryconvert(::Type{T}, x::T) where {T} = x
tryconvert(::Type{T}, x) where {T} =
    try
        convert(T, x)
    catch
        NOTIMPLEMENTED()
    end

CTryConvertRule_wrapref(o, ::Type{S}) where {S} = putresult(S(C.PyObjectRef(o)))
CTryConvertRule_trywrapref(o, ::Type{S}) where {S} =
    try
        putresult(S(C.PyObjectRef(o)))
    catch
        0
    end

@generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)

@generated _type_union_split(::Type{T}) where {T} = Vector{Type}(__type_union_split(T))

__type_union_split(T) = begin
    S = T
    vars = []
    while S isa UnionAll
        pushfirst!(vars, S.var)
        S = S.body
    end
    if S isa DataType || S isa TypeVar
        return [T]
    elseif S isa Union
        return [foldl((S,v)->UnionAll(v,S), vars, init=S) for S in [__type_union_split(S.a)..., __type_union_split(S.b)...]]
    elseif S == Union{}
        return []
    else
        @show S typeof(S)
        @assert false
    end
end

@generated _type_ub(::Type{T}) where {T} = begin
    S = T
    while S isa UnionAll
        S = S{S.var.ub}
    end
    S
end
@generated _type_lb(::Type{T}) where {T} = begin
    R = T
    while R isa UnionAll
        R = R.body
    end
    if R isa DataType
        S = T
        while S isa UnionAll
            S = S{S.var in R.parameters ? S.var.lb : S.var.ub}
        end
        S
    else
        _type_ub(T)
    end
end

function pointer_from_obj(o::T) where {T}
    if T.mutable
        c = o
        p = Base.pointer_from_objref(o)
    else
        c = Ref{Any}(o)
        p = unsafe_load(Ptr{Ptr{Cvoid}}(Base.pointer_from_objref(c)))
    end
    p, c
end

isnull(p::Ptr) = Ptr{Cvoid}(p) == C_NULL
isnull(p::UnsafePtr) = isnull(pointer(p))
ism1(x::T) where {T<:Number} = x == (zero(T) - one(T))

# Put something in here to keep it around forever
const CACHE = []
