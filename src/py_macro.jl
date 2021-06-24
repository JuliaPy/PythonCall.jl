const PY_MACRO_UNOPS = Dict(
    # operators
    :(+) => (pypos, true),
    :(-) => (pyneg, true),
    :(~) => (pyinv, true),
    # builtins
    :abs => (pyabs, true),
    :ascii => (pyascii, true),
    :bytes => (pybytes, true),
    :bool => (pybool, true),
    :complex => (pycomplex, true),
    :dict => (pydict, true),
    :dir => (pydir, true),
    :float => (pyfloat, true),
    :frozenset => (pyfrozenset, true),
    :repr => (pyrepr, true),
    :str => (pystr, true),
    # builtins converting to julia
    :jlascii => (x->pyascii(String,x), false),
    :jlbool => (pytruth, false),
    :jlbytes => (x->pybytes(Base.CodeUnits,x), false),
    :jlhash => (pyhash, false),
    :jllen => (pylen, false),
    :jlrepr => (x->pyrepr(String,x), false),
    :jlstr => (x->pystr(String,x), false),
    # jlcomplex
)

const PY_MACRO_BINOPS = Dict(
    # operators
    :(+) => (pyadd, true),
    :(-) => (pysub, true),
    :(*) => (pymul, true),
    :(/) => (pytruediv, true),
    :(÷) => (pyfloordiv, true),
    :(%) => (pymod, true),
    :(^) => (pypow, true),
    :(<<) => (pylshift, true),
    :(>>) => (pyrshift, true),
    :(&) => (pyand, true),
    :(|) => (pyor, true),
    :(⊻) => (pyxor, true),
    :(==) => (pyeq, true),
    :(!=) => (pyne, true),
    :(<=) => (pyle, true),
    :(< ) => (pylt, true),
    :(>=) => (pyge, true),
    :(> ) => (pygt, true),
    :(===) => (pyis, false),
    :(!==) => ((!) ∘ pyis, false),
    # builtins
    :pow => (pypow, true),
    :complex => (pycomplex, true),
    :delattr => (pydelattr, false),
    :divmod => (pydivmod, true),
    :getattr => (pygetattr, true),
    :hasattr => (pyhasattr, false),
    :issubclass => (pyissubclass, false),
    :isinstance => (pyisinstance, false),
    # builtins converting to julia
    # jlcomplex
)

const PY_MACRO_TERNOPS = Dict(
    :pow => (pypow, true),
)

mutable struct PyMacroState
    mod :: Module
    src :: LineNumberNode
    consts :: Vector{Any}
end

function py_macro_err(st, ex, msg=nothing)
    ex2 = ex isa Expr ? Expr(ex.head, [arg isa Expr ? :__ : arg for arg in ex.args]...) : ex
    message = "@py syntax error: $ex2"
    if msg !== nothing
        message *= ": " * msg
    end
    message *= "\n  at $(st.src.file):$(st.src.line)"
    error(message)
end

py_macro_assign(body, ans, ex) = push!(body, ans===nothing ? ex : :($ans = $ex))

py_macro_del(body, var, tmp) = if tmp; push!(body, :($pydel!($var))); end

function py_macro_lower(st, ex, body; ans=nothing)
    if ex isa Union{Nothing, String, Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, BigInt, Float16, Float32, Float64}
        x = pynew()
        push!(st.consts, :($pycopy!($x, $Py($ex))))
        py_macro_assign(body, ans, x)
        return false
    elseif ex isa Symbol
        if ex in BUILTINS
            py_macro_assign(body, ans, :($pybuiltins.$ex))
        else
            py_macro_assign(body, ans, ex)
        end
        return false
    # turn addition with more than three arguments into nested binary addition
    elseif @capture(ex, +(ax_, ay_, az_, args__))
        return py_macro_lower(st, foldl((x, y)->:($x+$y), (ax, ay, az, args...)), body, ans=ans)
    # turn multiplication with more than three arguments into nested binary multiplication
    elseif @capture(ex, *(ax_, ay_, az_, args__))
        return py_macro_lower(st, foldl((x, y)->:($x*$y), (ax, ay, az, args...)), body, ans=ans)
    elseif isexpr(ex, :call)
        af = ex.args[1]
        # is it a special operator?
        isop = false
        if haskey(PY_MACRO_UNOPS, af)
            isop = true
            op, t = PY_MACRO_UNOPS[af]
            if length(ex.args) == 2
                ax, = ex.args[2:end]
                @gensym x
                tx = py_macro_lower(st, ax, body, ans=x)
                py_macro_assign(body, ans, :($op($x)))
                py_macro_del(body, x, tx)
                return t
            end
        end
        if haskey(PY_MACRO_BINOPS, af)
            isop = true
            op, t = PY_MACRO_BINOPS[af]
            if length(ex.args) == 3
                ax, ay = ex.args[2:end]
                @gensym x y
                tx = py_macro_lower(st, ax, body, ans=x)
                ty = py_macro_lower(st, ay, body, ans=y)
                py_macro_assign(body, ans, :($op($x, $y)))
                py_macro_del(body, x, tx)
                py_macro_del(body, y, ty)
                return t
            end
        end
        if haskey(PY_MACRO_TERNOPS, af)
            isop = true
            op, t = PY_MACRO_TERNOPS[af]
            if length(ex.args) == 4
                ax, ay, az = ex.args[2:end]
                @gensym x y z
                tx = py_macro_lower(st, ax, body, ans=x)
                ty = py_macro_lower(st, ay, body, ans=y)
                tz = py_macro_lower(st, az, body, ans=y)
                py_macro_assign(body, ans, :($op($x, $y, $z)))
                py_macro_del(body, x, tx)
                py_macro_del(body, y, ty)
                py_macro_del(body, z, tz)
                return t
            end
        end
        # if it's an operator but we get this far, we used the operator wrong
        # if it's also a builtin, we allow calling it as a function
        # otherwise it's a syntax error
        isop && af ∉ BUILTINS && py_macro_err(st, ex)
        # disallow things like calling other julia infix operators like functions
        af isa Symbol && !Base.isidentifier(af) && py_macro_err(st, ex)
        # pycall
        aargs = []
        akwargs = []
        aparams = []
        for a in ex.args[2:end]
            if isexpr(a, :parameters)
                append!(aparams, a.args)
            elseif isexpr(a, :kw)
                push!(akwargs, a)
            else
                push!(aargs, a)
            end
        end
        append!(akwargs, aparams)
        if !isempty(akwargs)
            @gensym f args kwargs
            tf = py_macro_lower(st, af, body, ans=f)
            targs = py_macro_lower(st, Expr(:tuple, aargs...), body, ans=args)
            tkwargs = py_macro_lower(st, Expr(:braces, Expr(:parameters), akwargs...), body, ans=kwargs)
            py_macro_assign(body, ans, :($pycallargs($f, $args, $kwargs)))
            py_macro_del(body, f, tf)
            py_macro_del(body, args, targs)
            py_macro_del(body, kwargs, tkwargs)
        elseif !isempty(aargs)
            @gensym f args
            tf = py_macro_lower(st, af, body, ans=f)
            targs = py_macro_lower(st, Expr(:tuple, aargs...), body, ans=args)
            py_macro_assign(body, ans, :($pycallargs($f, $args)))
            py_macro_del(body, f, tf)
            py_macro_del(body, args, targs)
        else
            @gensym f
            tf = py_macro_lower(st, af, body, ans=f)
            py_macro_assign(body, ans, :($pycallargs($f)))
            py_macro_del(body, f, tf)
        end
        return true
    elseif isexpr(ex, :tuple)
        if any(isexpr(arg, :...) for arg in ex.args)
            py_macro_err(st, ex, "splatting into tuples not implemented")
        else
            py_macro_assign(body, ans, :($pynulltuple($(length(ex.args)))))
            @gensym a
            for (i, aa) in enumerate(ex.args)
                ta = py_macro_lower(st, aa, body, ans=a)
                push!(body, :($pytuple_setitem($ans, $(i-1), $a)))
                py_macro_del(body, a, ta)
            end
            return true
        end
    elseif isexpr(ex, :braces)
        # Like Python, we allow braces to be set or dict literals.
        #
        # The expression is a dict if either:
        # - it is empty ({})
        # - it has a parameters section (e.g. {x,y;})
        # - it has any assignment expressions (e.g. {k=v})
        # - it has any pair expressions (e.g. {k=>v})
        # and otherwise it is a set.
        #
        # e.g. these are sets: {x, y}, {x...}
        # e.g. these are dicts: {}, {x, y;}, {k=x}, {k=>x}
        if isempty(ex.args) || any(isexpr(arg, :parameters, :(=), :kw) || @capture(arg, k_:v_) for arg in ex.args)
            aargs = []
            for a in ex.args
                if isexpr(a, :parameters)
                    append!(aargs, a.args)
                else
                    push!(aargs, a)
                end
            end
            py_macro_assign(body, ans, :($pydict()))
            @gensym k v
            for aa in aargs
                if isexpr(aa, :...)
                    py_macro_err(st, aa, "splatting into dicts not implemented")
                elseif isexpr(aa, :kw, :(=))
                    ak, av = aa.args
                    ak isa Symbol || py_macro_err(st, aa, "key of `key=value` must be a symbol - did you mean `key:value`?")
                    tk = py_macro_lower(st, string(ak), body, ans=k)
                    tv = py_macro_lower(st, av, body, ans=v)
                    push!(body, :($pydict_setitem($ans, $k, $v)))
                    py_macro_del(body, k, tk)
                    py_macro_del(body, v, tv)
                elseif @capture(aa, ak_ : av_)
                    tk = py_macro_lower(st, ak, body, ans=k)
                    tv = py_macro_lower(st, av, body, ans=v)
                    push!(body, :($pydict_setitem($ans, $k, $v)))
                    py_macro_del(body, k, tk)
                    py_macro_del(body, v, tv)
                else
                    py_macro_err(st, aa, "this kind of dict entry not implemented")
                end
            end
            return true
        else
            py_macro_assign(body, ans, :($pyset()))
            @gensym a
            for aa in ex.args
                if isexpr(aa, :...)
                    py_macro_err(st, ex, "splatting into sets not implemented")
                end
                ta = py_macro_lower(st, aa, body, ans=a)
                push!(body, :($pyset_add($ans, $a)))
                py_macro_del(body, a, ta)
            end
            return true
        end
    end
    py_macro_err(st, ex)
end

function py_macro(ex, mod, src)
    body = []
    @gensym ans
    st = PyMacroState(mod, src, [])
    py_macro_lower(st, ex, body; ans=ans)
    if !isempty(st.consts)
        initconsts = Ref(true)
        pushfirst!(body, Expr(:if, :($initconsts[]), Expr(:block, st.consts..., :($initconsts[] = false))))
    end
    Expr(:block, body..., ans)
end

macro py(ex)
    esc(py_macro(ex, __module__, __source__))
end
export @py
