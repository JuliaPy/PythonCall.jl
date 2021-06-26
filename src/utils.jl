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

end
