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

size_to_fstrides(elsz::Integer, sz::Integer...) =
    isempty(sz) ? () : (elsz, size_to_fstrides(elsz * sz[1], sz[2:end]...)...)

size_to_cstrides(elsz::Integer, sz::Integer...) =
    isempty(sz) ? () : (size_to_cstrides(elsz * sz[end], sz[1:end-1]...)..., elsz)

isfcontiguous(o::AbstractArray) = strides(o) == size_to_fstrides(1, size(o)...)
isccontiguous(o::AbstractArray) = strides(o) == size_to_cstrides(1, size(o)...)

pybufferformat(::Type{T}) where {T} =
    T == Int8 ? "=b" :
    T == UInt8 ? "=B" :
    T == Int16 ? "=h" :
    T == UInt16 ? "=H" :
    T == Int32 ? "=i" :
    T == UInt32 ? "=I" :
    T == Int64 ? "=q" :
    T == UInt64 ? "=Q" :
    T == Float16 ? "=e" :
    T == Float32 ? "=f" :
    T == Float64 ? "=d" :
    T == Complex{Float16} ? "=Ze" :
    T == Complex{Float32} ? "=Zf" :
    T == Complex{Float64} ? "=Zd" :
    T == Bool ? "?" :
    T == Ptr{Cvoid} ? "P" :
    T == C.PyObjectRef ? "O" :
    if isstructtype(T) && isconcretetype(T) && allocatedinline(T)
        n = fieldcount(T)
        flds = []
        for i = 1:n
            nm = fieldname(T, i)
            tp = fieldtype(T, i)
            push!(flds, string(pybufferformat(tp), nm isa Symbol ? ":$nm:" : ""))
            d =
                (i == n ? sizeof(T) : fieldoffset(T, i + 1)) -
                (fieldoffset(T, i) + sizeof(tp))
            @assert d ≥ 0
            d > 0 && push!(flds, "$(d)x")
        end
        string("T{", join(flds, " "), "}")
    else
        "$(sizeof(T))x"
    end

pybufferformat_to_type(fmt::AbstractString) =
    fmt == "b" ? Cchar :
    fmt == "B" ? Cuchar :
    fmt == "h" ? Cshort :
    fmt == "H" ? Cushort :
    fmt == "i" ? Cint :
    fmt == "I" ? Cuint :
    fmt == "l" ? Clong :
    fmt == "L" ? Culong :
    fmt == "q" ? Clonglong :
    fmt == "Q" ? Culonglong :
    fmt == "e" ? Float16 :
    fmt == "f" ? Cfloat :
    fmt == "d" ? Cdouble :
    fmt == "?" ? Bool :
    fmt == "P" ? Ptr{Cvoid} :
    fmt == "O" ? C.PyObjectRef :
    fmt == "=e" ? Float16 :
    fmt == "=f" ? Float32 : fmt == "=d" ? Float64 : error("not implemented: $(repr(fmt))")

pytypestrdescr_to_type(ts::String, descr) = begin
    # byte swapped?
    bsc = ts[1]
    bs =
        bsc == '<' ? !islittleendian() :
        bsc == '>' ? islittleendian() :
        bsc == '|' ? false : error("endianness character not supported: $ts")
    bs && error("byte-swapping not supported: $ts")
    # element type
    etc = ts[2]
    if etc == 'b'
        sz = parse(Int, ts[3:end])
        sz == sizeof(Bool) && return Bool
        error("bool of this size not supported: $ts")
    elseif etc == 'i'
        sz = parse(Int, ts[3:end])
        sz == 1 && return Int8
        sz == 2 && return Int16
        sz == 4 && return Int32
        sz == 8 && return Int64
        sz == 16 && return Int128
        error("signed int of this size not supported: $ts")
    elseif etc == 'u'
        sz = parse(Int, ts[3:end])
        sz == 1 && return UInt8
        sz == 2 && return UInt16
        sz == 4 && return UInt32
        sz == 8 && return UInt64
        sz == 16 && return UInt128
        error("unsigned int of this size not supported: $ts")
    elseif etc == 'f'
        sz = parse(Int, ts[3:end])
        sz == 2 && return Float16
        sz == 4 && return Float32
        sz == 8 && return Float64
        error("float of this size not supported: $ts")
    elseif etc == 'c'
        sz = parse(Int, ts[3:end])
        sz == 4 && return Complex{Float16}
        sz == 8 && return Complex{Float32}
        sz == 16 && return Complex{Float64}
        error("complex of this size not supported: $ts")
    elseif etc == 'O'
        return C.PyObjectRef
    else
        error("type not supported: $ts")
    end
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
