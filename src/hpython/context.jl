if DEBUG
    Context(c::C.Context) = Context(c, Builtins(), Dict{Int, C.PyPtr}(), 1)
else
    Context(c::C.Context) = Context(c, Builtins())
end

Context(; params...) = Context(C.Context(; params...))

@inline Base.getproperty(ctx::Context, k::Symbol) = hasfield(Context, k) ? getfield(ctx, k) : Builtin{k}(ctx)

function Base.show(io::IO, ctx::Context)
    show(io, typeof(ctx))
    print(io, "(")
    show(io, ctx._c)
    print(io, ", ...)")
end

function Base.show(io::IO, b::Builtin)
    show(io, typeof(b))
    print(io, "(...)")
end

function Base.show(io::IO, b::Builtins)
    show(io, typeof(b))
    print(io, "(")
    n = 0
    for k in fieldnames(typeof(b))
        v = getfield(b, k)
        if v !== PyNULL
            n += 1
            if n > 1
                print(io, ", ")
            end
            print(io, k, "=")
            show(io, getfield(b, k))
        end
    end
    print(io, ")")
end
