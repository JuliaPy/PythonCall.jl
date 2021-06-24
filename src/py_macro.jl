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
    :int => (pyint, true),
    :iter => (pyiter, true),
    :list => (pylist, true),
    :range => (pyrange, true),
    :repr => (pyrepr, true),
    :set => (pyset, true),
    :slice => (pyslice, true),
    :str => (pystr, true),
    :tuple => (pytuple, true),
    :type => (pytype, true),
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
    :range => (pyrange, true),
    :slice => (pyslice, true),
    # builtins converting to julia
    # jlcomplex
)

const PY_MACRO_TERNOPS = Dict(
    :pow => (pypow, true),
    :range => (pyrange, true),
    :setattr => (pysetattr, false),
    :slice => (pyslice, true),
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

py_macro_assign(body, ans, ex) = push!(body, :($ans = $ex))

py_macro_del(body, var, tmp) = if tmp; push!(body, :($pydel!($var))); end

function py_macro_lower(st, body, ans, ex; flavour=:expr)

    # string literal
    if ex isa String
        x = pynew()
        push!(st.consts, :($pycopy!($x, $pystr_intern!($pystr($ex)))))
        py_macro_assign(body, ans, x)
        return false

    # scalar literals
    elseif ex isa Union{Nothing, String, Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, BigInt, Float16, Float32, Float64}
        x = pynew()
        push!(st.consts, :($pycopy!($x, $Py($ex))))
        py_macro_assign(body, ans, x)
        return false

    # x
    elseif ex isa Symbol
        if ex in BUILTINS
            py_macro_assign(body, ans, :($pybuiltins.$ex))
        else
            py_macro_assign(body, ans, ex)
        end
        return false

    # x:y:z
    elseif flavour==:index && @capture(ex, ax_:ay_:az_)
        @gensym x y z
        tx = py_macro_lower(st, body, x, ax===:_ ? :None : ax)
        ty = py_macro_lower(st, body, y, ay===:_ ? :None : ay)
        tz = py_macro_lower(st, body, z, az===:_ ? :None : az)
        py_macro_assign(body, ans, :($pyslice($x, $y, $z)))
        py_macro_del(body, x, tx)
        py_macro_del(body, y, ty)
        py_macro_del(body, z, tz)
        return true

    # x:y
    elseif flavour==:index && @capture(ex, ax_:ay_)
        @gensym x y
        tx = py_macro_lower(st, body, x, ax===:_ ? :None : ax)
        ty = py_macro_lower(st, body, y, ay===:_ ? :None : ay)
        py_macro_assign(body, ans, :($pyslice($x, $y)))
        py_macro_del(body, x, tx)
        py_macro_del(body, y, ty)
        return true

    # x + y + z + ...
    elseif @capture(ex, +(ax_, ay_, az_, args__))
        return py_macro_lower(st, body, ans, foldl((x, y)->:($x+$y), (ax, ay, az, args...)))

    # x * y * z * ...
    elseif @capture(ex, *(ax_, ay_, az_, args__))
        return py_macro_lower(st, body, ans, foldl((x, y)->:($x*$y), (ax, ay, az, args...)))

    # f(args...; kwargs...)
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
                tx = py_macro_lower(st, body, x, ax)
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
                tx = py_macro_lower(st, body, x, ax)
                ty = py_macro_lower(st, body, y, ay)
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
                tx = py_macro_lower(st, body, x, ax)
                ty = py_macro_lower(st, body, y, ay)
                tz = py_macro_lower(st, body, z, az)
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
            tf = py_macro_lower(st, body, f, af)
            targs = py_macro_lower(st, body, args, Expr(:tuple, aargs...))
            tkwargs = py_macro_lower(st, body, kwargs, Expr(:braces, Expr(:parameters), akwargs...))
            py_macro_assign(body, ans, :($pycallargs($f, $args, $kwargs)))
            py_macro_del(body, f, tf)
            py_macro_del(body, args, targs)
            py_macro_del(body, kwargs, tkwargs)
        elseif !isempty(aargs)
            @gensym f args
            tf = py_macro_lower(st, body, f, af)
            targs = py_macro_lower(st, body, args, Expr(:tuple, aargs...))
            py_macro_assign(body, ans, :($pycallargs($f, $args)))
            py_macro_del(body, f, tf)
            py_macro_del(body, args, targs)
        else
            @gensym f
            tf = py_macro_lower(st, body, f, af)
            py_macro_assign(body, ans, :($pycallargs($f)))
            py_macro_del(body, f, tf)
        end
        return true

    # (...)
    elseif isexpr(ex, :tuple)
        if any(isexpr(arg, :...) for arg in ex.args)
            py_macro_err(st, ex, "splatting into tuples not implemented")
        else
            py_macro_assign(body, ans, :($pynulltuple($(length(ex.args)))))
            @gensym a
            for (i, aa) in enumerate(ex.args)
                ta = py_macro_lower(st, body, a, aa, flavour = flavour==:index ? :index : :expr)
                push!(body, :($pytuple_setitem($ans, $(i-1), $a)))
                py_macro_del(body, a, ta)
            end
            return true
        end

    # [...]
    elseif isexpr(ex, :vect)
        if any(isexpr(arg, :...) for arg in ex.args)
            py_macro_err(st, ex, "splatting into tuples not implemented")
        else
            py_macro_assign(body, ans, :($pynulllist($(length(ex.args)))))
            @gensym a
            for (i, aa) in enumerate(ex.args)
                ta = py_macro_lower(st, body, a, aa)
                push!(body, :($pylist_setitem($ans, $(i-1), $a)))
                py_macro_del(body, a, ta)
            end
            return true
        end

    # {...}
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
                    tk = py_macro_lower(st, body, k, string(ak))
                    tv = py_macro_lower(st, body, v, av)
                    push!(body, :($pydict_setitem($ans, $k, $v)))
                    py_macro_del(body, k, tk)
                    py_macro_del(body, v, tv)
                elseif @capture(aa, ak_ : av_)
                    tk = py_macro_lower(st, body, k, ak)
                    tv = py_macro_lower(st, body, v, av)
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
                ta = py_macro_lower(st, body, a, aa)
                push!(body, :($pyset_add($ans, $a)))
                py_macro_del(body, a, ta)
            end
            return true
        end

    # x.k
    elseif isexpr(ex, :.)
        ax, ak = ex.args
        @gensym x k
        tx = py_macro_lower(st, body, x, ax)
        if ak isa QuoteNode && ak.value isa Symbol
            tk = py_macro_lower(st, body, k, string(ak.value))
        else
            tk = py_macro_lower(st, body, k, ak)
        end
        py_macro_assign(body, ans, :($pygetattr($x, $k)))
        py_macro_del(body, x, tx)
        py_macro_del(body, k, tk)
        return true

    # x[k]
    elseif @capture(ex, ax_[ak__])
        @gensym x k
        tx = py_macro_lower(st, body, x, ax)
        if length(ak) == 1
            tk = py_macro_lower(st, body, k, ak[1]; flavour=:index)
        else
            tk = py_macro_lower(st, body, k, Expr(:tuple, ak...); flavour=:index)
        end
        py_macro_assign(body, ans, :($pygetitem($x, $k)))
        py_macro_del(body, x, tx)
        py_macro_del(body, k, tk)
        return true

    # x = y
    elseif @capture(ex, ax_ = ay_)
        ty = py_macro_lower(st, body, ans, ay)
        py_macro_lower_assign(st, body, ax, ans)
        return ty

    # @del x, y, ...
    elseif @capture(ex, @del (args__,))

        for arg in args

            # @del x
            if arg isa Symbol
                if arg in BUILTINS
                    py_macro_err(st, ex, "can't delete a builtin")
                else
                    push!(body, :($pydel!($arg::$Py)))
                end

            # @del x.k
            elseif isexpr(arg, :.)
                ax, ak = ex.args
                @gensym x k
                tx = py_macro_lower(st, body, x, ax)
                if ak isa QuoteNode && ak.value isa Symbol
                    tk = py_macro_lower(st, body, k, string(ak.value))
                else
                    tk = py_macro_lower(st, body, k, ak)
                end
                push!(body, :($pydelattr($x, $k)))
                py_macro_del(body, x, tx)
                py_macro_del(body, k, tk)

            # @del x[k]
            elseif @capture(arg, ax_[ak__])
                @gensym x k
                tx = py_macro_lower(st, body, x, ax)
                if length(ak) == 1
                    tk = py_macro_lower(st, body, k, ak[1], flavour=:index)
                else
                    tk = py_macro_lower(st, body, k, Expr(:tuple, ak...), flavour=:index)
                end
                push!(body, :($pydelitem($x, $k)))
                py_macro_del(body, x, tx)
                py_macro_del(body, k, tk)

            else
                py_macro_err(st, ex, "@del argument must be a variable, reference or property")
            end
        end
        py_macro_assign(body, ans, nothing)
        return false

    # @del x
    elseif @capture(ex, @del arg_)
        return py_macro_lower(st, body, ans, :(@del ($arg,)))

    # @jl x
    elseif @capture(ex, @jl ax_)
        y = py_macro_lower_jl(st, ax)
        py_macro_assign(body, ans, y)
        return false

    # begin; ...; end
    elseif isexpr(ex, :block)
        if isempty(ex.args)
            py_macro_assign(body, ans, nothing)
            return false
        else
            @gensym a
            for (i, aa) in enumerate(ex.args)
                if aa isa LineNumberNode
                    st.src = aa
                    push!(body, aa)
                    if i == length(ex.args)
                        py_macro_assign(body, ans, nothing)
                        return false
                    end
                elseif i == length(ex.args)
                    ta = py_macro_lower(st, body, ans, aa)
                    return ta
                else
                    ta = py_macro_lower(st, body, a, aa)
                    py_macro_del(body, a, ta)
                end
            end
            @assert false
        end

    # if x; ...; end
    elseif isexpr(ex, :if, :elseif)
        if length(ex.args) == 2
            return py_macro_lower(st, body, ans, Expr(ex.head, ex.args..., :None))
        elseif length(ex.args) == 3
            ax, ay, az = ex.args
            @gensym x
            py_macro_lower_bool(st, body, x, ax)
            body2 = []
            ty = py_macro_lower(st, body2, ans, ay)
            body3 = []
            tz = py_macro_lower(st, body3, ans, az)
            t = ty || tz
            if t
                ty || push!(body2, :($ans = $Py($ans)))
                tz || push!(body3, :($ans = $Py($ans)))
            end
            push!(body, Expr(:if, x, Expr(:block, body2...), Expr(:block, body3...)))
            return t
        end

    # while x; ...; end
    elseif isexpr(ex, :while)
        ax, ay = ex.args
        @gensym x y
        body2 = []
        py_macro_lower_bool(st, body2, x, ax)
        push!(body2, Expr(:if, :(!$x), :(break)))
        ty = py_macro_lower(st, body2, y, ay)
        py_macro_del(body2, y, ty)
        push!(body, Expr(:while, true, Expr(:block, body2...)))
        py_macro_assign(body, ans, nothing)
        return false

    # for x in y; ...; end
    elseif @capture(ex, for ax_ in ay_; az_; end)
        @gensym y i v z
        ty = py_macro_lower(st, body, y, ay)
        push!(body, :($i = $pyiter($y)))
        py_macro_del(body, y, ty)
        body2 = []
        push!(body2, :($v = $pynext($i)))
        push!(body2, Expr(:if, :($ispynull($v)), Expr(:block, :($pydel!($v)), :(break))))
        py_macro_lower_assign(st, body2, ax, v)
        py_macro_del(body2, v, true)
        tz = py_macro_lower(st, body2, z, az)
        py_macro_del(body2, z, tz)
        push!(body, Expr(:while, true, Expr(:block, body2...)))
        py_macro_del(body, i, true)
        py_macro_assign(body, ans, nothing)
        return false

    # import ...
    elseif isexpr(ex, :import)
        for aa in ex.args
            if isexpr(aa, :as)
                am, av = aa.args
                av isa Symbol || py_macro_err(st, av)
                py_macro_lower_import(st, body, av, am)
            elseif isexpr(aa, :.)
                @gensym m
                if length(aa.args) > 1
                    tm = py_macro_lower_import(st, body, m, aa)
                    py_macro_del(body, m, tm)
                end
                py_macro_lower_import(st, body, aa.args[1], Expr(:., aa.args[1]))
            elseif isexpr(aa, :(:))
                @gensym m
                py_macro_lower_import(st, body, m, aa.args[1])
                for ak in aa.args[2:end]
                    if isexpr(ak, :as)
                        ak, av = ak.args
                        av isa Symbol || py_macro_err(st, av)
                        py_macro_lower_nested_getattr(st, body, av, m, ak)
                    elseif isexpr(ak, :.)
                        py_macro_lower_nested_getattr(st, body, ak.args[end], m, ak)
                    else
                        py_macro_err(st, ak)
                    end
                end
                py_macro_del(body, m, true)
            else
                py_macro_err(st, aa)
            end
        end
        py_macro_assign(body, ans, nothing)
        return false
    end

    py_macro_err(st, ex)
end

function py_macro_lower_nested_getattr(st, body, ans, x, ks)
    @gensym m v
    isexpr(ks, :.) || py_macro_err(st, ks)
    length(ks.args) > 0 || py_macro_err(st, ks)
    py_macro_lower(st, body, ans, :($x.$(ks.args[1])))
    for k in ks.args[2:end]
        py_macro_lower(st, body, v, :($ans.$k))
        py_macro_del(body, ans, true)
        py_macro_assign(body, ans, v)
    end
end

function py_macro_lower_import(st, body, ans, ex)
    @gensym m
    isexpr(ex, :.) || py_macro_err(st, ex)
    tm = py_macro_lower(st, body, m, join(ex.args, "."))
    py_macro_assign(body, ans, :($pyimport($m)))
    py_macro_del(body, m, tm)
end

function py_macro_lower_bool(st, body, ans, ex)
    # TODO: special cases to avoid round-tripping bools from Julia to Python to Julia
    @gensym x
    t = py_macro_lower(st, body, x, ex)
    py_macro_assign(body, ans, :($pytruth($x)))
    return
end

function py_macro_lower_assign(st, body, lhs, rhs::Symbol)
    if lhs isa Symbol
        push!(body, :($lhs = $Py($rhs)))
    elseif @capture(lhs, ax_[ak__])
        @gensym x k
        tx = py_macro_lower(st, body, x, ax)
        if length(ak) == 1
            tk = py_macro_lower(st, body, k, ak[1], flavour=:index)
        else
            tk = py_macro_lower(st, body, k, Expr(:tuple, ak...), flavour=:index)
        end
        push!(body, :($pysetitem($x, $k, $rhs)))
        py_macro_del(body, x, tx)
        py_macro_del(body, k, tk)
    elseif isexpr(lhs, :.)
        ax, ak = lhs.args
        @gensym x k
        tx = py_macro_lower(st, body, x, ax)
        if ak isa QuoteNode && ak.value isa Symbol
            tk = py_macro_lower(st, body, k, string(ak.value))
        else
            tk = py_macro_lower(st, body, k, ak)
        end
        push!(body, :($pysetattr($x, $k, $rhs)))
        py_macro_del(body, x, tx)
        py_macro_del(body, k, tk)
    else
        py_macro_err(st, lhs, "assignment")
    end
    return
end

function py_macro_lower_jl(st, ex)
    # TODO: expand out macros
    if @capture(ex, @py ax_)
        @gensym x
        body = []
        py_macro_lower(st, body, x, ax)
        return Expr(:block, body..., x)
    elseif ex isa Expr
        return Expr(ex.head, [py_macro_lower_jl(st, aa) for aa in ex.args]...)
    else
        return ex
    end
end

function py_macro(ex, mod, src)
    body = []
    @gensym ans
    st = PyMacroState(mod, src, [])
    py_macro_lower(st, body, ans, ex)
    if !isempty(st.consts)
        initconsts = Ref(true)
        pushfirst!(body, Expr(:if, :($initconsts[]), Expr(:block, st.consts..., :($initconsts[] = false))))
    end
    pushfirst!(body, src)
    Expr(:block, body..., ans)
end

macro py(ex)
    esc(py_macro(ex, __module__, __source__))
end
export @py
