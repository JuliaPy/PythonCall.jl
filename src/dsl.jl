using MacroTools

PYDSL_OPS = Dict(
    (:length, 1) => :PyObject_Length,
    (:+, 1) => :PyNumber_Positive,
    (:+, 2) => :PyNumber_Add,
    (:-, 1) => :PyNumber_Negative,
    (:-, 2) => :PyNumber_Subtract,
    (:*, 2) => :PyNumber_Multiply,
    (:/, 2) => :PyNumber_TrueDivide,
    (:÷, 2) => :PyNumber_FloorDivide,
    (:div, 2) => :PyNumber_FloorDivide,
    (:%, 2) => :PyNumber_Remainder,
    (:mod, 2) => :PyNumber_Remainder,
    (:abs, 1) => :PyNumber_Absolute,
    (:~, 1) => :PyNumber_Invert,
    (:^, 2) => :PyNumber_Power,
    (:powermod, 3) => :PyNumber_Power,
    (:<<, 2) => :PyNumber_Lshift,
    (:>>, 2) => :PyNumber_Rshift,
    (:|, 2) => :PyNumber_Or,
    (:&, 2) => :PyNumber_And,
    (:⊻, 2) => :PyNumber_Xor,
    (:xor, 2) => :PyNumber_Xor,
)

PYEXPR_HEADS = Set([
    :PyVar,
    :PyObject_From,
    :PyObject_GetAttr,
    :PyObject_GetItem,
    :PyTuple,
    :PyNumber_Add,
    :PyNumber_Positive,
    :PyNumber_Subtract,
    :PyNumber_Negative,
    :PyNumber_Multiply,
    :PyNumber_TrueDivide,
    :PyNumber_FloorDivide,
    :PyNumber_Remainder,
    :PyNumber_Absolute,
    :PyNumber_Invert,
    :PyNumber_Power,
    :PyNumber_Lshift,
    :PyNumber_Rshift,
    :PyNumber_Or,
    :PyNumber_And,
    :PyNumber_Xor,
    :PyObject_Call,
    :PyImport_Import,
    :PyImport_ImportModuleLevelObject,
])

ispyexpr(ex) = ex isa Expr && (
    ex.head in PYEXPR_HEADS
    || (ex.head == :block && !isempty(ex.args) && ispyexpr(last(ex.args)))
    || (ex.head == :(=) && length(ex.args)==2 && ispyexpr(ex.args[2]))
)

aspyexpr(ex) = ispyexpr(ex) ? ex : Expr(:PyObject_From, ex)

nopyexpr(ex) = ispyexpr(ex) ? Expr(:PyObject_Convert, ex, PyObject) : ex

@kwdef struct PyDSLInterpretState
    ispyvar :: Dict{Symbol, Bool} = Dict{Symbol, Bool}()
    __module__ :: Module
end

ispyvar(v::Symbol, st::PyDSLInterpretState) = get!(st.ispyvar, v, false)
addpyvar(v::Symbol, st::PyDSLInterpretState) = @assert get!(st.ispyvar, v, true)

function pydsl_interpret_lhs(ex, st::PyDSLInterpretState, ispy::Bool)
    if ex isa Symbol
        @assert get!(st.ispyvar, ex, ispy) == ispy
        ex
    else
        error("pydsl_interpret: assignment is only implemented for single variables")
    end
end

function pydsl_interpret(ex, st::PyDSLInterpretState)
    if ex isa Symbol
        if ispyvar(ex, st)
            Expr(:PyVar, ex)
        else
            ex
        end
    elseif ex isa Expr
        head = ex.head
        args = ex.args
        nargs = length(args)

        # lhs::T
        if head == :(::) && nargs == 2
            lhs, T = args
            if T === :Py
                if lhs isa Expr && lhs.head == :tuple
                    xs2 = [pydsl_interpret(x, st) for x in lhs.args]
                    Expr(:PyTuple, xs2...)
                else
                    lhs2 = pydsl_interpret(lhs, st)
                    if ispyexpr(lhs2)
                        lhs2
                    else
                        Expr(:PyObject_From, lhs2)
                    end
                end
            else
                lhs2 = pydsl_interpret(lhs, st)
                T2 = pydsl_interpret(T, st)
                if ispyexpr(lhs2)
                    Expr(:PyObject_Convert, lhs2, T2)
                else
                    Expr(:(::), lhs2, T2)
                end
            end

        # begin; ...; end
        elseif head == :block
            args2 = [pydsl_interpret(arg, st) for arg in args]
            Expr(:block, args2...)

        # lhs = rhs
        elseif head == :(=) && nargs == 2
            lhs, rhs = args
            rhs2 = pydsl_interpret(rhs, st)
            lhs2 = pydsl_interpret_lhs(lhs, st, ispyexpr(rhs2))
            Expr(:(=), lhs2, rhs2)

        # x.k
        elseif head == :. && nargs == 2
            x, k = args
            x2 = pydsl_interpret(x, st)
            k2 = pydsl_interpret(k, st)
            if ispyexpr(x2)
                Expr(:PyObject_GetAttr, x2, k2)
            else
                Expr(:., x2, k2)
            end

        # x[ks...]
        elseif head == :ref && nargs ≥ 1
            x, ks... = args
            x2 = pydsl_interpret(x, st)
            ks2 = [pydsl_interpret(k, st) for k in ks]
            if ispyexpr(x2)
                if length(ks2) == 1
                    Expr(:PyObject_GetItem, x2, ks2[1])
                else
                    Expr(:PyObject_GetItem, x2, Expr(:PyTuple, ks2...))
                end
            else
                Expr(:ref, x2, ks2...)
            end

        # op(xs...)
        elseif head == :call && nargs ≥ 1 && haskey(PYDSL_OPS, (args[1], nargs-1))
            _, xs... = args
            xs2 = [pydsl_interpret(x, st) for x in xs]
            if any(ispyexpr, xs2)
                Expr(PYDSL_OPS[(args[1], nargs-1)], xs2...)
            else
                Expr(:call, args[1], xs2...)
            end

        # +(xs...)
        elseif head == :call && nargs ≥ 1 && args[1] == :+
            _, xs... = args
            xs2 = [pydsl_interpret(x, st) for x in xs]
            if any(ispyexpr, xs2)
                if length(xs2) == 1
                    Expr(:PyNumber_Positive, xs2[1])
                else
                    ret = xs2[1]
                    for x in xs2[2:end]
                        ret = Expr(:PyNumber_Add, ret, x)
                    end
                    ret
                end
            else
                Expr(:call, :+, xs2...)
            end

        # *(xs...)
        elseif head == :call && nargs ≥ 1 && args[1] == :*
            _, xs... = args
            xs2 = [pydsl_interpret(x, st) for x in xs]
            if any(ispyexpr, xs2)
                ret = xs2[1]
                for x in xs2[2:end]
                    ret = Expr(:PyNumber_Multiply, ret, x)
                end
                ret
            else
                Expr(:call, :*, xs2...)
            end

        # f(...)
        elseif head == :call && nargs ≥ 1
            args2 = [pydsl_interpret(arg, st) for arg in args]
            if ispyexpr(args2[1])
                Expr(:PyObject_Call, args2...)
            else
                Expr(:call, args2...)
            end
        elseif head == :kw && nargs == 2
            k, v = args
            v2 = pydsl_interpret(v, st)
            Expr(:kw, k, v2)
        elseif head == :parameters
            args = [pydsl_interpret(arg, st) for arg in args]
            Expr(:parameters, args...)

        # @pyimport ...
        elseif head == :macrocall && args[1] == Symbol("@pyimport")
            res = Expr(:block)
            function modname(ex)
                if ex isa AbstractString
                    return convert(String, ex)
                elseif ex isa Symbol
                    return string(ex)
                elseif ex isa QuoteNode
                    return modname(ex.value)
                elseif ex isa Expr && ex.head == :. && length(ex.args) == 2
                    return "$(modname(ex.args[1])).$(modname(ex.args[2]))"
                else
                    error("syntax error: expecting Python module: $ex")
                end
            end
            function modnamevarname(ex)
                if ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[1] == :(=>)
                    _, mname, vname = ex.args
                    vname isa Symbol || error("syntax error")
                    modname(mname), vname, true
                else
                    mname = modname(ex)
                    vname = Symbol(split(mname, ".")[1])
                    modname(mname), vname, false
                end
            end
            for arg in args[3:end]
                if arg isa Expr && arg.head == :call && length(arg.args) == 3 && arg.args[1] == :(:)
                    _, mname, attrs = arg.args
                    mname = modname(mname)
                    if attrs isa Expr && attrs.head == :tuple
                        attrs = [modnamevarname(attr) for attr in attrs.args]
                    else
                        attrs = [modnamevarname(attrs)]
                    end
                    anames = [a for (a,_,_) in attrs]
                    vnames = [v for (_,v,_) in attrs]
                    for v in vnames
                        addpyvar(v, st)
                    end
                    push!(res.args, :(($(vnames...),) = $(Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyTuple, anames...), 0))))
                else
                    mname, vname, named = modnamevarname(arg)
                    addpyvar(vname, st)
                    push!(res.args, :($vname = $(Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyObject_From, PyLazyObject(named && '.' in mname ? ("*",) : ())), 0))))
                end
            end
            res

        # @m(...)
        elseif ex.head == :macrocall && nargs ≥ 2
            pydsl_interpret(macroexpand(st.__module__, ex, recursive=false), st)

        else
            error("pydsl_interpret: not implemented: Expr($(repr(head)), ...)")
        end
    else
        return ex
    end
end

macro pydsl_interpret(ex)
    QuoteNode(pydsl_interpret(ex, PyDSLInterpretState(__module__=__module__)))
end

export @pydsl_interpret

struct PyExpr
    ex
    var :: Symbol
    tmp :: Bool
end

@kwdef struct PyDSLLowerState
    tmpvars_unused :: Set{Symbol} = Set{Symbol}()
    tmpvars_used :: Set{Symbol} = Set{Symbol}()
    pyvars :: Set{Symbol} = Set{Symbol}()
    pyerrblocks :: Vector{Expr} = Vector{Expr}()
end

function pydsl_tmpvar(st::PyDSLLowerState)
    v = isempty(st.tmpvars_unused) ? gensym("pytmp") : pop!(st.tmpvars_unused)
    push!(st.tmpvars_used, v)
    v
end

function pydsl_discard(st::PyDSLLowerState, v::Symbol)
    @assert v in st.tmpvars_used
    delete!(st.tmpvars_used, v)
    push!(st.tmpvars_unused, v)
end

function pydsl_discard(st::PyDSLLowerState, v::PyExpr)
    if ex.tmp
        pydsl_discard(st, ex.var)
    end
end

function pydsl_free(st::PyDSLLowerState, v::Symbol)
    pydsl_discard(st, v)
    quote
        $(C.Py_DecRef)($v)
        $v = $CPyPtr(0)
    end
end

function pydsl_free(st::PyDSLLowerState, ex::PyExpr)
    if ex.tmp
        pydsl_free(st, ex.var)
    else
        Expr(:block)
    end
end

function pydsl_errblock(st::PyDSLLowerState, ignorevars...)
    ex = Expr(:block, [:($(C.Py_DecRef)($v)) for v in st.tmpvars_used if v ∉ ignorevars]...)
    push!(st.pyerrblocks, ex)
    ex
end

function pydsl_lower_inner(ex, st::PyDSLLowerState)
    if ex isa Expr
        head = ex.head
        args = ex.args
        nargs = length(args)
        if head == :PyVar
            nargs == 1 || error("syntax error")
            var, = args
            var isa Symbol || error("syntax error")
            PyExpr(Expr(:block), var, false)
        elseif head == :PyObject_From
            nargs == 1 || error("syntax error")
            x, = args
            if x isa String
                x2 = PyInternedString(x)
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $t = $(C.PyObject_From)($x2)
                end, t, true)
            elseif x isa Bool
                t = gensym("pytmp")
                PyExpr(quote
                    $t = $(x ? C.Py_True : C.Py_False)()
                end, t, false)
            elseif x isa Nothing || x === :nothing
                t = gensym("pytmp")
                PyExpr(quote
                    $t = $(C.Py_None)()
                end, t, false)
            elseif x isa Number
                x2 = PyLazyObject(x)
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $t = $(C.PyObject_From)($x2)
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            else
                x2 = pydsl_lower_inner(x, st)
                if x2 isa PyExpr
                    x2
                else
                    t = pydsl_tmpvar(st)
                    PyExpr(quote
                        $t = $(C.PyObject_From)($x2) :: $CPyPtr
                        if $t == $CPyPtr(0)
                            $(pydsl_errblock(st, t))
                        end
                    end, t, true)
                end
            end
        elseif head == :PyObject_Convert
            nargs == 2 || error("syntax error")
            x, T = args
            x2 = pydsl_lower_inner(x, st)
            T2 = pydsl_lower_inner(T, st)
            T2 isa PyExpr && error("syntax error")
            x2 isa PyExpr || error("syntax error")
            err = gensym("err")
            if T2 isa Expr
                Tv = gensym("T")
                quote
                    $Tv = $T2
                    $(x2.ex)
                    $err = $(C.PyObject_Convert)($(x2.var), $T2)
                    $(pydsl_free(st, x2))
                    if $err == -1
                        $(pydsl_errblock(st))
                    end
                    $(takeresult)($Tv)
                end
            else
                quote
                    $(x2.ex)
                    $err = $(C.PyObject_Convert)($(x2.var), $T2)
                    $(pydsl_free(st, x2))
                    if $err == -1
                        $(pydsl_errblock(st))
                    end
                    $(takeresult)($T2)
                end
            end
        elseif head == :PyObject_GetAttr
            nargs == 2 || error("syntax error")
            x, k = args
            ispyexpr(x) || error("syntax error")
            x2 = pydsl_lower_inner(x, st) :: PyExpr
            if k isa QuoteNode && k.value isa Symbol
                k = string(k.value)
            end
            k2 = pydsl_lower_inner(aspyexpr(k), st) :: PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(x2.ex)
                $(k2.ex)
                $t = $(C.PyObject_GetAttr)($(x2.var), $(k2.var))
                $(pydsl_free(st, x2))
                $(pydsl_free(st, k2))
                if $t == $CPyPtr(0)
                    $(pydsl_errblock(st, t))
                end
            end, t, true)
        elseif head == :PyObject_Call
            nargs ≥ 1 || error("syntax error")
            f, allargs... = args
            f2 = pydsl_lower_inner(aspyexpr(f), st) :: PyExpr
            varargs = []
            kwargs = []
            for arg in allargs
                if arg isa Expr && arg.head == :parameters
                    append!(kwargs, arg.args)
                elseif arg isa Expr && arg.head == :kw && length(arg.args) == 2
                    push!(kwargs, arg)
                else
                    push!(varargs, arg)
                end
            end
            if !isempty(kwargs)
                error("kwargs not implemented")
            elseif !isempty(varargs)
                varargs2 = pydsl_lower_inner(Expr(:PyTuple, varargs...), st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(f2.ex)
                    $(varargs2.ex)
                    $t = $(C.PyObject_CallObject)($(f2.var), $(varargs2.var))
                    $(pydsl_free(st, f2))
                    $(pydsl_free(st, varargs2))
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            else
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(f2.ex)
                    $t = $(C.PyObject_CallObject)($(f2.var), $C_NULL)
                    $(pydsl_free(st, f2))
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            end
        elseif head == :PyTuple
            if any(ex -> ex isa Expr && ex.head == :..., args)
                error("splatting into a tuple not implemented")
            else
                t = pydsl_tmpvar(st)
                err = gensym("err")
                PyExpr(quote
                    $t = $(C.PyTuple_New)($(length(args)))
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                    $([
                        begin
                            arg2 = pydsl_lower_inner(aspyexpr(arg), st) :: PyExpr
                            quote
                                $(arg2.ex)
                                $(C.Py_IncRef)($(arg2.var))
                                $err = $(C.PyTuple_SetItem)($t, $(i-1), $(arg2.var))
                                $(pydsl_free(st, arg2))
                                if $err == -1
                                    $(pydsl_errblock(st))
                                end
                            end
                        end
                        for (i,arg) in enumerate(args)
                    ]...)
                end, t, true)
            end
        elseif head in (:PyNumber_Positive, :PyNumber_Negative, :PyNumber_Absolute, :PyNumber_Invert, :PyImport_Import)
            nargs == 1 || error("syntax error")
            x, = args
            x2 = pydsl_lower_inner(aspyexpr(x), st) :: PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(x2.ex)
                $t = $(getproperty(C, head))($(x2.var))
                $(pydsl_free(st, x2))
                if $t == $CPyPtr(0)
                    $(pydsl_errblock(st, t))
                end
            end, t, true)
        elseif head in (:PyNumber_Add, :PyNumber_Subtract, :PyNumber_Multiply, :PyNumber_TrueDivide, :PyNumber_FloorDivide, :PyNumber_Remainder, :PyNumber_Lshift, :PyNumber_Rshift, :PyNumber_Or, :PyNumber_Xor, :PyNumber_And, :PyObject_GetItem)
            nargs == 2 || error("syntax error")
            x, y = args
            x2 = pydsl_lower_inner(aspyexpr(x), st) :: PyExpr
            y2 = pydsl_lower_inner(aspyexpr(y), st) :: PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(x2.ex)
                $(y2.ex)
                $t = $(getproperty(C, head))($(x2.var), $(y2.var))
                $(pydsl_free(st, x2))
                $(pydsl_free(st, y2))
                if $t == $CPyPtr(0)
                    $(pydsl_errblock(st, t))
                end
            end, t, true)
        elseif head == :PyNumber_Power
            if nargs == 2
                x, y = args
                x2 = pydsl_lower_inner(aspyexpr(x), st) :: PyExpr
                y2 = pydsl_lower_inner(aspyexpr(y), st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(x2.ex)
                    $(y2.ex)
                    $t = $(C.PyNumber_Power)($(x2.var), $(y2.var), $(C.Py_None)())
                    $(pydsl_free(st, x2))
                    $(pydsl_free(st, y2))
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            elseif nargs == 3
                x, y, z = args
                x2 = pydsl_lower_inner(aspyexpr(x), st) :: PyExpr
                y2 = pydsl_lower_inner(aspyexpr(y), st) :: PyExpr
                z2 = pydsl_lower_inner(aspyexpr(z), st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(x2.ex)
                    $(y2.ex)
                    $(z2.ex)
                    $t = $(C.PyNumber_Power)($(x2.var), $(y2.var), $(z2.var))
                    $(pydsl_free(st, x2))
                    $(pydsl_free(st, y2))
                    $(pydsl_free(st, z2))
                    if $t == $CPyPtr(0)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            else
                error("syntax error")
            end
        elseif head == :PyObject_Length
            nargs == 1 || error("syntax error")
            x, = args
            x2 = pydsl_lower_inner(aspyexpr(x), st) :: PyExpr
            len = gensym("len")
            quote
                $(x2.ex)
                $len = $(C.PyObject_Length)($(x2.var))
                $(pydsl_free(st, x2))
                if $len == -1
                    $(pydsl_errblock(st))
                end
                $len
            end
        elseif head == :PyImport_ImportModuleLevelObject
            nargs == 5 || error("syntax error")
            mod, gls, lcs, frs, lvl = args
            lvl isa Int || error("syntax error")
            mod2 = pydsl_lower_inner(aspyexpr(mod), st) :: PyExpr
            gls2 = pydsl_lower_inner(aspyexpr(gls), st) :: PyExpr
            lcs2 = pydsl_lower_inner(aspyexpr(lcs), st) :: PyExpr
            frs2 = pydsl_lower_inner(aspyexpr(frs), st) :: PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(mod2.ex)
                $(gls2.ex)
                $(lcs2.ex)
                $(frs2.ex)
                $t = $(C.PyImport_ImportModuleLevelObject)($(mod2.var), $(gls2.var), $(lcs2.var), $(frs2.var), $lvl)
                $(pydsl_free(st, mod2))
                $(pydsl_free(st, gls2))
                $(pydsl_free(st, lcs2))
                $(pydsl_free(st, frs2))
                if $t == $CPyPtr(0)
                    $(pydsl_errblock(st, t))
                end
            end, t, true)
        elseif head == :block
            res = Expr(:block)
            for (i, arg) in enumerate(args)
                arg2 = pydsl_lower_inner(arg, st)
                if arg2 isa PyExpr
                    if i == length(args)
                        # last expression is a PyExpr, so whole block is a PyExpr
                        @assert ispyexpr(ex)
                        push!(res.args, arg2.ex)
                        return PyExpr(res, arg2.var, arg2.tmp)
                    else
                        push!(res.args, quote
                            $(arg2.ex)
                            $(pydsl_free(st, arg2))
                        end)
                    end
                else
                    push!(res.args, arg2)
                end
            end
            @assert !ispyexpr(ex)
            res
        elseif head == :call
            args2 = [pydsl_lower_inner(nopyexpr(arg), st) for arg in args]
            Expr(:call, args2...)
        elseif head == :(=) && nargs == 2
            lhs, rhs = args
            if lhs isa Symbol
                rhs2 = pydsl_lower_inner(rhs, st)
                if rhs2 isa PyExpr
                    push!(st.pyvars, lhs)
                    PyExpr(quote
                        $(rhs2.ex)
                        $(C.Py_DecRef)($lhs)
                        $lhs = $(rhs2.var)
                        $(C.Py_IncRef)($lhs)
                    end, rhs2.var, rhs2.tmp)
                else
                    Expr(:(=), lhs, rhs2)
                end
            else
                error("pydsl_lower: not implemented: assignment to anything other than a single variable")
            end
        else
            error("pydsl_lower: not implemented: Expr($(repr(head)), ...)")
        end
    else
        ex
    end
end

function pydsl_lower(ex; onpyerror, onjlerror)
    ex = nopyexpr(ex)
    st = PyDSLLowerState()
    ex = pydsl_lower_inner(ex, st)
    @assert isempty(st.tmpvars_used)
    @assert !ispyexpr(ex)
    inits = [:($v :: $CPyPtr = $CPyPtr(0)) for v in st.pyvars]
    decrefs = [:($(C.Py_DecRef)($v)) for v in st.pyvars]
    for block in st.pyerrblocks
        prepend!(block.args, decrefs)
        push!(block.args, onpyerror)
    end
    if onjlerror === :impossible
        ans = gensym("ans")
        quote
            let
                $(inits...)
                $ans = $(ex)
                $(decrefs...)
                $ans
            end
        end
    else
        isfreed = gensym("freed")
        for block in st.pyerrblocks
            pushfirst!(block.args, :($isfreed = true))
        end
        quote
            let
                $isfreed = false
                $(inits...)
                try
                    $((ex isa Expr && ex.head == :block ? ex.args : [ex])...)
                catch exc
                    $((onjlerror isa Expr && onjlerror.head == :block ? onjlerror.args : [onjlerror])...)
                finally
                    if !$isfreed
                        $(decrefs...)
                    end
                end
            end
        end
    end
end

macro pydsl(ex, kwargs...)
    onpyerror = :($pythrow())
    onjlerror = :($rethrow())
    interpret = true
    for kwarg in kwargs
        kwarg isa Expr && kwarg.head == :(=) && length(kwarg.args) == 2 || error("syntax error: $(kwarg)")
        k, v = kwarg.args
        if k == :onpyerror
            onpyerror = v
        elseif k == :onjlerror
            onjlerror = v
        elseif k == :interpret
            v isa Bool || error("interpret argument must be a Bool")
            interpret = v
        else
            error("invalid kwarg: $k")
        end
    end
    # interpret
    if interpret
        ex = pydsl_interpret(ex, PyDSLInterpretState(__module__=__module__))
    end
    # lower
    esc(pydsl_lower(ex; onpyerror=onpyerror, onjlerror=onjlerror))
end

export @pydsl

macro pydsl_expand(args...)
    ex = @eval @macroexpand1 @pydsl $(args...)
    ex = MacroTools.striplines(ex)
    ex = MacroTools.flatten(ex)
    ex = MacroTools.unresolve(ex)
    ex = MacroTools.resyntax(ex)
    ex = MacroTools.gensym_ids(ex)
    QuoteNode(ex)
end

export @pydsl_expand
