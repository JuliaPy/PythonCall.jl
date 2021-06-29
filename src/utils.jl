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

end
