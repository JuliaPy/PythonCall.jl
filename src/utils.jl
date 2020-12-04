size_to_fstrides(elsz::Integer, sz::Integer...) =
    isempty(sz) ? () : (elsz, size_to_fstrides(elsz * sz[1], sz[2:end]...)...)

size_to_cstrides(elsz::Integer, sz::Integer...) =
    isempty(sz) ? () : (size_to_cstrides(elsz * sz[end], sz[1:end-1]...)..., elsz)

isfcontiguous(o::AbstractArray) = strides(o) == size_to_fstrides(1, size(o)...)
isccontiguous(o::AbstractArray) = strides(o) == size_to_cstrides(1, size(o)...)

# TODO: make this better: e.g. views are immutable structures, but should be considered mutable arrays
ismutablearray(x::AbstractArray) = !isimmutable(x)
ismutablearray(x::SubArray) = ismutablearray(parent(x))

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
    if isstructtype(T) && isconcretetype(T) && Base.allocatedinline(T)
        n = fieldcount(T)
        flds = []
        for i in 1:n
            nm = fieldname(T, i)
            tp = fieldtype(T, i)
            push!(flds, string(pybufferformat(tp), nm isa Symbol ? ":$nm:" : ""))
            d = (i==n ? sizeof(T) : fieldoffset(T, i+1)) - (fieldoffset(T, i) + sizeof(tp))
            @assert d≥0
            d>0 && push!(flds, "$(d)x")
        end
        string("T{", join(flds, " "), "}")
    else
        "$(Base.aligned_sizeof(T))x"
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
    fmt == "O" ? CPyObjRef :
    fmt == "=e" ? Float16 :
    fmt == "=f" ? Float32 :
    fmt == "=d" ? Float64 :
    error("not implemented: $(repr(fmt))")

### TYPE UTILITIES

struct PyConvertFail end

tryconvert(::Type{T}, x) where {T} =
try
    convert(T, x)
catch
    PyConvertFail()
end
tryconvert(::Type{T}, x::T) where {T} = x
tryconvert(::Type{T}, x::PyConvertFail) where {T} = x

@generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)

@generated function _type_flatten_tuple(::Type{T}) where {T<:Tuple}
    S = T
    vars = []
    while !isa(S, DataType)
        push!(vars, S.var)
        S = S.body
    end
    Tuple{[foldr(UnionAll, vars; init=P) for P in S.parameters]...}
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
