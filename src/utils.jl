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

end
