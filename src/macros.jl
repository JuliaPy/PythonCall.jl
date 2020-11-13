"""
    @pyeval ex

Evaluate the given Python expression `ex`.

Julia values may be interpolated using `\$(julia_expr)` syntax.
"""
macro pyeval(src)
    parts = String[]
    escapes = Dict{String,Any}()
    if src isa AbstractString
        push!(parts, src)
    elseif src isa Expr && src.head == :string
        n = 0
        for ex in src.args
            if ex isa AbstractString
                push!(parts, ex)
            else
                n += 1
                v = "__jl_escape_$n"
                escapes[v] = ex
                push!(parts, "($v)")
            end
        end
    else
        error("invalid source string")
    end
    newsrc = join(parts)
    :($pyeval($newsrc, pydict($([Expr(:kw, Symbol(k), esc(ex)) for (k,ex) in escapes]...))))
end
export @pyeval

"""
    @pyexecute ex [retvars]

Execute the given Python code `ex`.

Julia values may be interpolated using `\$(julia_expr)` syntax.

If `retvars` is given, it is the name of a variable (or tuple of names) to return from the global scope of the expression. Each name may be annotated with a type, and `pyconvert` will be called to perform the conversion. As a special case `(*)` will return the entire global scope as a dictionary.
"""
macro pyexec(src, ret=nothing)
    parts = String[]
    escapes = Dict{String, Any}()
    if src isa AbstractString
        push!(parts, src)
    elseif src isa Expr && src.head == :string
        n = 0
        for ex in src.args
            if ex isa AbstractString
                push!(parts, ex)
            else
                n += 1
                v = "__jl_escape_$n"
                escapes[v] = ex
                push!(parts, "($v)")
            end
        end
    else
        error("invalid source string")
    end
    function mkretex(ex)
        t = PyObject
        if ex isa Expr && ex.head == :(::)
            t = esc(ex.args[2])
            ex = ex.args[1]
        end
        ex isa Symbol || error("invalid return value `$(ex)`")
        :(pyconvert($t, d[$(string(ex))]))
    end
    if ret === nothing
        retex = nothing
    elseif ret === :*
        retex = :d
    elseif ret isa Expr && ret.head == :tuple
        retex = Expr(:tuple, map(mkretex, ret.args)...)
    else
        retex = mkretex(ret)
    end
    newsrc = join(parts)
    :(let d=pydict($([Expr(:kw, Symbol(k), esc(ex)) for (k,ex) in escapes]...))
        $pyexec($newsrc, d)
        $retex
    end)
end
export @pyexec
