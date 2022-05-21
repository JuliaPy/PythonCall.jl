module Utils

    function explode_union(T)
        @nospecialize T

        # unpack unionall
        S = T
        vars = []
        while S isa UnionAll
            pushfirst!(vars, S.var)
            S = S.body
        end

        if S isa Union
            Us = Any[explode_union(S.a)..., explode_union(S.b)...]
            Any[foldl((body, var) -> UnionAll(var, body), vars, init=U) for U in Us]
        elseif S == Union{}
            Any[]
        else
            Any[T]
        end
    end

    """
        pointer_from_obj(x)

    Returns `(p, c)` where `Base.pointer_from_objref(p) === x`.

    The pointer remains valid provided the object `c` is not garbage collected.
    """
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

    """
        ExtraNewline(x)

    An object that displays the same as `x` but with an extra newline in text/plain.
    """
    struct ExtraNewline{T}
        value :: T
    end
    Base.show(io::IO, m::MIME, x::ExtraNewline) = show(io, m, x.value)
    Base.show(io::IO, m::MIME"text/csv", x::ExtraNewline) = show(io, m, x.value)
    Base.show(io::IO, m::MIME"text/tab-separated-values", x::ExtraNewline) = show(io, m, x.value)
    Base.show(io::IO, m::MIME"text/plain", x::ExtraNewline) = (show(io, m, x.value); println(io))
    Base.showable(m::MIME, x::ExtraNewline) = showable(m, x.value)

    const ALL_MIMES = [
        "text/plain",
        "text/html",
        "text/markdown",
        "text/json",
        "text/latex",
        "text/xml",
        "text/csv",
        "application/javascript",
        "application/pdf",
        "application/ogg",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/gif",
        "image/webp",
        "image/tiff",
        "image/bmp",
        "audio/aac",
        "audio/mpeg",
        "audio/ogg",
        "audio/opus",
        "audio/webm",
        "audio/wav",
        "audio/midi",
        "audio/x-midi",
        "video/mpeg",
        "video/ogg",
        "video/webm",
    ]

    function mimes_for(x)
        @nospecialize x
        # default mimes we always try
        mimes = copy(ALL_MIMES)
        # look for mimes on show methods for this type
        for meth in methods(show, Tuple{IO, MIME, typeof(x)}).ms
            mimetype = meth.sig.parameters[3]
            mimetype isa DataType || continue
            mime = string(mimetype.parameters[1])
            push!(mimes, mime)
        end
        return mimes
    end

    @generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)

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

    @generated _promote_type_bounded(::Type{S}, ::Type{T}, ::Type{B}) where {S,T,B} = begin
        S <: B || error("require S <: B")
        T <: B || error("require T <: B")
        if B isa Union
            return Union{_promote_type_bounded(typeintersect(S, B.a), typeintersect(T, B.a), B.a), _promote_type_bounded(typeintersect(S, B.b), typeintersect(T, B.b), B.b)}
        else
            R = promote_type(S, T)
            if R <: B
                return R
            else
                R = typeintersect(typejoin(S, T), B)
                if R <: B
                    return R
                else
                    return B
                end
            end
        end
    end

    @generated _promote_type_bounded(::Type{T1}, ::Type{T2}, ::Type{T3}, ::Type{B}) where {T1,T2,T3,B} =
        _promote_type_bounded(_promote_type_bounded(T1, T2, B), T3, B)

    # TODO: what is the best way?
    ismutablearray(x::Array) = true
    ismutablearray(x::AbstractArray) = begin
        p = parent(x)
        p === x ? false : ismutablearray(p)
    end

    islittleendian() = Base.ENDIAN_BOM == 0x04030201 ? true : Base.ENDIAN_BOM == 0x01020304 ? false : error()

    isflagset(flags, mask) = (flags & mask) == mask

    size_to_fstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
        isempty(sz) ? () : (elsz, size_to_fstrides(elsz * sz[1], sz[2:end])...)

    size_to_cstrides(elsz::Integer, sz::Tuple{Vararg{Integer}}) =
        isempty(sz) ? () : (size_to_cstrides(elsz * sz[end], sz[1:end-1])..., elsz)

    struct StaticString{T,N} <: AbstractString
        codeunits :: NTuple{N,T}
        StaticString{T,N}(codeunits::NTuple{N,T}) where {T,N} = new{T,N}(codeunits)
    end

    function Base.print(io::IO, x::StaticString)
        cs = collect(x.codeunits)
        i = findfirst(==(0), cs)
        print(io, transcode(String, i===nothing ? cs : cs[1:i-1]))
    end

    function Base.convert(::Type{StaticString{T,N}}, x::AbstractString) where {T,N}
        cs = collect(transcode(T, convert(String, x)))
        length(cs) > N && throw(InexactError(:convert, StaticString{T,N}, x))
        while length(cs) < N
            push!(cs, 0)
        end
        StaticString{T,N}(NTuple{N,T}(cs))
    end

    StaticString{T,N}(x::AbstractString) where {T,N} = convert(StaticString{T,N}, x)

    function Base.iterate(x::StaticString, st::Union{Nothing,Tuple}=nothing)
        if st === nothing
            s = String(x)
            z = iterate(s)
        else
            s, st0 = st
            z = iterate(s, st0)
        end
        if z === nothing
            nothing
        else
            c, newst0 = z
            (c, (s, newst0))
        end
    end
end
