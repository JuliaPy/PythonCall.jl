using MacroTools

struct PyDSLSyntaxError <: Exception
    msg :: String
    src :: LineNumberNode
end

Base.showerror(io::IO, e::PyDSLSyntaxError) = begin
    println(io, "PyDSLSyntaxError: ", e.msg)
    print(io,  "   @ ", e.src.file, ":", e.src.line)
end

PYDSL_OPS = Dict(
    :PyObject_Repr => (1, any, :py),
    :PyObject_Str => (1, any, :py),
    :PyObject_Bytes => (1, any, :py),
    :PyObject_ASCII => (1, any, :py),
    :PyObject_Type => (1, any, :py),
    :PyObject_Dir => (1, any, :py),
    :PyObject_GetIter => (1, any, :py),
    :PyNumber_Positive => (1, any, :py),
    :PyNumber_Negative => (1, any, :py),
    :PyNumber_Absolute => (1, any, :py),
    :PyNumber_Invert => (1, any, :py),
    :PyNumber_Long => (1, any, :py),
    :PyNumber_Float => (1, any, :py),
    :PyNumber_Index => (1, any, :py),
    :PyImport_Import => (1, any, :py),
    :PySequence_List => (1, any, :py),
    :PySequence_Tuple => (1, any, :py),
    :PyObject_IsTrue => (1, any, :bool),
    :PyObject_Not => (1, any, :bool),
    :PyObject_Length => (1, any, :int),
    :PyObject_Hash => (1, any, :int),
    :PyNumber_Add => (2, any, :py),
    :PyNumber_Subtract => (2, any, :py),
    :PyNumber_Multiply => (2, any, :py),
    :PyNumber_Power => (2:3, any, :py),
    :PyNumber_TrueDivide => (2, any, :py),
    :PyNumber_FloorDivide => (2, any, :py),
    :PyNumber_Remainder => (2, any, :py),
    :PyNumber_Lshift => (2, any, :py),
    :PyNumber_Rshift => (2, any, :py),
    :PyNumber_Or => (2, any, :py),
    :PyNumber_And => (2, any, :py),
    :PyNumber_Xor => (2, any, :py),
    :PyNumber_InPlaceAdd => (2, any, :py),
    :PyNumber_InPlaceSubtract => (2, any, :py),
    :PyNumber_InPlaceMultiply => (2, any, :py),
    :PyNumber_InPlacePower => (2:3, any, :py),
    :PyNumber_InPlaceTrueDivide => (2, any, :py),
    :PyNumber_InPlaceFloorDivide => (2, any, :py),
    :PyNumber_InPlaceRemainder => (2, any, :py),
    :PyNumber_InPlaceLshift => (2, any, :py),
    :PyNumber_InPlaceRshift => (2, any, :py),
    :PyNumber_InPlaceOr => (2, any, :py),
    :PyNumber_InPlaceAnd => (2, any, :py),
    :PyNumber_InPlaceXor => (2, any, :py),
    :PyObject_Eq => (2, any, :py),
    :PyObject_Ne => (2, any, :py),
    :PyObject_Lt => (2, any, :py),
    :PyObject_Le => (2, any, :py),
    :PyObject_Gt => (2, any, :py),
    :PyObject_Ge => (2, any, :py),
    :PyObject_GetItem => (2, x->x[1], :py),
    :PyObject_GetAttr => (2, x->x[1], :py),
    :PyObject_Is => (2, any, :bool),
    :PyObject_IsNot => (2, any, :bool),
    :PyObject_IsInstance => (2, all, :bool),
    :PyObject_IsSubclass => (2, all, :bool),
    :PyObject_HasAttr => (2, x->x[1], :bool),
    :PySequence_Contains => (2, x->x[1], :bool),
    :PyObject_In => (2, x->x[2], :bool),
    :PyObject_NotIn => (2, x->x[2], :bool),
    :PyObject_DelItem => (2, x->x[1], :void),
    :PyObject_DelAttr => (2, x->x[1], :void),
    :PyObject_SetItem => (3, x->x[1], :void),
    :PyObject_SetAttr => (3, x->x[1], :void),
)

PYDSL_OPS_ALIAS = Dict(
    :PyPtr => (:PyPtr, 1, any),
    :length => (:PyObject_Length, 1, any),
    :Bool => (:PyObject_IsTrue, 1, any),
    :! => (:PyObject_Not, 1, any),
    :+ => (:PyNumber_Positive, 1, any),
    :+ => (:PyNumber_Add, 2, any),
    :- => (:PyNumber_Negative, 1, any),
    :- => (:PyNumber_Subtract, 2, any),
    :* => (:PyNumber_Multiply, 2, any),
    :/ => (:PyNumber_TrueDivide, 2, any),
    :÷ => (:PyNumber_FloorDivide, 2, any),
    :div => (:PyNumber_FloorDivide, 2, any),
    :% => (:PyNumber_Remainder, 2, any),
    :mod => (:PyNumber_Remainder, 2, any),
    :abs => (:PyNumber_Absolute, 1, any),
    :~ => (:PyNumber_Invert, 1, any),
    :^ => (:PyNumber_Power, 2, any),
    :powermod => (:PyNumber_Power, 3, any),
    :<< => (:PyNumber_Lshift, 2, any),
    :>> => (:PyNumber_Rshift, 2, any),
    :| => (:PyNumber_Or, 2, any),
    :& => (:PyNumber_And, 2, any),
    :⊻ => (:PyNumber_Xor, 2, any),
    :xor => (:PyNumber_Xor, 2, any),
    :(===) => (:PyObject_Is, 2, any),
    :(≡) => (:PyObject_Is, 2, any),
    :(!==) => (:PyObject_IsNot, 2, any),
    :(≢) => (:PyObject_IsNot, 2, any),
    :(==) => (:PyObject_Eq, 2, any),
    :(!=) => (:PyObject_Ne, 2, any),
    :(≠) => (:PyObject_Ne, 2, any),
    :(<) => (:PyObject_Lt, 2, any),
    :(<=) => (:PyObject_Le, 2, any),
    :(≤) => (:PyObject_Le, 2, any),
    :(>) => (:PyObject_Gt, 2, any),
    :(>=) => (:PyObject_Ge, 2, any),
    :(≥) => (:PyObject_Ge, 2, any),
    :(isa) => (:PyObject_IsInstance, 2, all),
    :(<:) => (:PyObject_IsSubclass, 2, all),
    :(in) => (:PyObject_In, 2, x->x[2]),
    :(∈) => (:PyObject_In, 2, x->x[2]),
    :(∉) => (:PyObject_NotIn, 2, x->x[2]),
)

PYDSL_IOPS = Dict(
    :(+=) => :PyNumber_InPlaceAdd,
    :(-=) => :PyNumber_InPlaceSubtract,
    :(*=) => :PyNumber_InPlaceMultiply,
    :(^=) => :PyNumber_InPlacePower,
    :(/=) => :PyNumber_InPlaceTrueDivide,
    :(÷=) => :PyNumber_InPlaceFloorDivide,
    :(%=) => :PyNumber_InPlaceRemainder,
    :(<<=) => :PyNumber_InPlaceLshift,
    :(>>=) => :PyNumber_InPlaceRshift,
    :(|=) => :PyNumber_InPlaceOr,
    :(&=) => :PyNumber_InPlaceAnd,
    :(⊻=) => :PyNumber_InPlaceXor,
)

PYEXPR_HEADS = Set([
    :PyVar,
    :PyExternObject,
    :PyTuple,
    :PyList,
    :PyDict,
    :PySet,
    :PyKwargs,
    :PyBlock,
    :PyAssign,
    :PyInPlaceAssign,
    :PyIf,
    :PyAnd,
    :PyOr,
    :PyObject_From,
    :PyObject_Call,
    :PyImport_ImportModuleLevelObject,
    :PyTypeAssert,
    [k for (k,(_,_,t)) in PYDSL_OPS if t == :py]...,
])

ispyexpr(ex) = ex isa Expr && ex.head in PYEXPR_HEADS

aspyexpr(ex) = ispyexpr(ex) ? ex : Expr(:PyObject_From, ex)

nopyexpr(ex) = ispyexpr(ex) ? Expr(:PyObject_Convert, ex, PythonCall.PyObject) : ex

ignorepyexpr(ex) = ispyexpr(ex) ? Expr(:PyIgnore, ex) : ex

truthpyexpr(ex) = ispyexpr(ex) ? Expr(:PyTruth, ex) : ex

@kwdef mutable struct PyDSLInterpretState
    ispyvar :: Dict{Symbol, Bool} = Dict{Symbol, Bool}()
    __module__ :: Module
    __source__ :: LineNumberNode
end

pydsl_syntax_error(st::PyDSLInterpretState, msg="") = throw(PyDSLSyntaxError(msg, st.__source__))

ispyvar(v::Symbol, st::PyDSLInterpretState, dflt::Bool=false) = get!(st.ispyvar, v, dflt)
addpyvar(v::Symbol, st::PyDSLInterpretState) = @assert get!(st.ispyvar, v, true)

pydsl_dropline(x::LineNumberNode) = isfile(string(x.file)) && realpath(string(x.file)) == realpath(@__FILE__)
pydsl_srcstr(x::LineNumberNode) = x.file === nothing ? "<unknown>" : "$(x.file):$(x.line)"
pydsl_srcstr(x::PyDSLInterpretState) = pydsl_srcstr(x.__source__)

function pydsl_interpret_lhs(ex, st::PyDSLInterpretState, ispy::Bool)
    if ex isa Symbol
        if ex == :_
            ispy ? Expr(:PyLHSIgnore) : ex
        elseif ispyvar(ex, st, ispy)
            Expr(:PyLHSVar, ex)
        else
            ex
        end
    elseif @capture(ex, x_[args__])
        x2 = pydsl_interpret(x, st)
        args2 = [pydsl_interpret(x, st) for x in args]
        k2 = length(args2)==1 ? args2[1] : Expr(:PyTuple, args2...)
        if ispyexpr(x2)
            Expr(:PyLHSItem, x2, k2)
        else
            :($x2[$(args2...)])
        end
    elseif @capture(ex, x_.k_)
        x2 = pydsl_interpret(x, st)
        if k isa Symbol
            k2 = ispyexpr(x2) ? Expr(:PyObject_From, string(k)) : k
        else
            k2 = pydsl_interpret(k, st)
        end
        if ispyexpr(x2)
            Expr(:PyLHSAttr, x2, k2)
        else
            :($x2.$k2)
        end
    else
        pydsl_syntax_error(st, "assignment to $ex")
    end
end

function pydsl_interpret_typeassert(x, T, st)
    x2 = pydsl_interpret(x, st)
    if T === :Py
        ispyexpr(x2) || pydsl_syntax_error(st, "not a Python expression: $x")
        x2
    else
        T2 = pydsl_interpret(T, st)
        if ispyexpr(T2) && ispyexpr(x2)
            Expr(:PyTypeAssert, x2, T2)
        elseif !ispyexpr(T2) && !ispyexpr(x2)
            :($x2 :: $T2)
        else
            pydsl_syntax_error(st, "mixed Python/Julia types: $(:($x::$T))")
        end
    end
end

function pydsl_interpret_call(f, args, kwargs, st; interpret_f=true)
    argispy = Bool[]
    args2 = []
    kwargs2 = []
    if kwargs !== nothing
        for arg in kwargs
            if arg isa Expr && arg.head == :kw && length(arg.args) == 2
                k, v = arg.args
                v2 = pydsl_interpret(v, st)
                push!(kwargs2, Expr(:kw, k, v2))
            else
                arg2 = pydsl_interpret(arg, st)
                anypy |= ispyexpr(arg2)
                push!(kwargs2, arg2)
            end
        end
    end
    for arg in args
        if arg isa Expr && arg.head == :kw && length(arg.args) == 2
            k, v = arg.args
            v2 = pydsl_interpret(v, st)
            push!(kwargs2, Expr(:kw, k, v2))
        else
            arg2 = pydsl_interpret(arg, st)
            push!(args2, arg2)
            push!(argispy, ispyexpr(arg2))
        end
    end
    anypy = any(argispy)
    na = length(args2)
    nk = length(kwargs2)
    # addition
    if interpret_f && anypy && f === :+
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        @assert na > 0
        if na == 1
            Expr(:PyNumber_Positive, args2...)
        else
            foldl((x,y)->Expr(:PyNumber_Add, x, y), args2)
        end
    # multiplication
    elseif interpret_f && anypy && f === :*
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na > 1 || pydsl_syntax_error(st, "`$f` requires at least 2 arguments")
        foldl((x,y)->Expr(:PyNumber_Multiply, x, y), args2)
    # operators
    elseif interpret_f &&  anypy && haskey(PYDSL_OPS_ALIAS, f)
        op, opna, match = PYDSL_OPS_ALIAS[f]
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na in opna || pydsl_syntax_error(st, "`$f` requires $opna arguments")
        match(argispy) || pydsl_syntax_error(st, "`$f` used ambiguously")
        Expr(op, args2...)
    elseif interpret_f && anypy && haskey(PYDSL_OPS, f)
        opna, match, _ = PYDSL_OPS[f]
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na in opna || pydsl_syntax_error(st, "`$f` requires $opna arguments")
        match(argispy) || pydsl_syntax_error(st, "`$f` used ambiguously")
        Expr(f, args2...)
    # Py(x), PyExt(x), PyExtX(x), PyExtB(x), PyExtBX(x)
    elseif interpret_f && f isa Symbol && f in (:Py, :PyExt, :PyExtX, :PyExtB, :PyExtBX)
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na == 1 || pydsl_syntax_error(st, "`$f` requires 1 argument")
        x2 = args2[1]
        if f == :Py
            if ispyexpr(x2)
                x2
            else
                Expr(:PyObject_From, x2)
            end
        else
            Expr(:PyExternObject, x2, f in (:PyExtB, :PyExtBX), f in (:PyExt, :PyExtB))
        end
    # convert(T, x::Py)
    elseif interpret_f && anypy && f === :convert
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na == 2 || pydsl_syntax_error(st, "`$f` requires 2 arguments")
        ispyexpr(args2[2]) || pydsl_syntax_error(st, "`$f` used ambiguously")
        Expr(:PyObject_Convert, args2[2], args2[1])
    # PyObject_Convert(x::Py, T)
    elseif interpret_f && anypy && f === :PyObject_Convert
        nk == 0 || pydsl_syntax_error(st, "`$f` does not accept keyword arguments")
        na == 2 || pydsl_syntax_error(st, "`$f` requires 2 arguments")
        ispyexpr(args2[1]) || pydsl_syntax_error(st, "`$f` used ambiguously")
        Expr(:PyObject_Convert, args2[1], args2[2])
    else
        f2 = interpret_f ? pydsl_interpret(f, st) : f
        if ispyexpr(f2)
            if !isempty(kwargs2)
                Expr(:PyObject_Call, f2, Expr(:PyTuple, args2...), Expr(:PyKwargs, kwargs2...))
            elseif !isempty(args2)
                Expr(:PyObject_Call, f2, Expr(:PyTuple, args2...))
            else
                Expr(:PyObject_Call, f2)
            end
        elseif kwargs === nothing
            :($f2($(args2...)))
        else
            :($f2($(args2...); $(kwargs2...)))
        end
    end
end

function modname(st, ex)
    if ex isa AbstractString
        return convert(String, ex)
    elseif ex isa Symbol
        return string(ex)
    elseif ex isa QuoteNode
        return modname(st, ex.value)
    elseif ex isa Expr && ex.head == :.
        return join([modname(st, arg) for arg in ex.args], ".")
    else
        pydsl_syntax_error(st, "expecting Python module, got `$ex`")
    end
end

function modnamevarname(st, ex)
    if @capture(ex, (vname_ = mname_) | (mname_ => vname_))
        vname isa Symbol || pydsl_syntax_error(st, "in `@pyimport var=module`, `var` must be a variable name, got `$vname`")
        modname(st, mname), vname, true
    elseif ex isa Expr && ex.head == :as && length(ex.args) == 2
        mname, vname = ex.args
        vname isa Symbol || pydsl_syntax_error(st, "in `@pyimport var=module`, `var` must be a variable name, got `$vname`")
        modname(st, mname), vname, true
    else
        mname = modname(st, ex)
        vname = Symbol(split(mname, ".")[1])
        mname, vname, false
    end
end

function pydsl_interpret(ex, st::PyDSLInterpretState)

    if ex isa Symbol
        if ex == :_
            ex
        elseif ispyvar(ex, st)
            Expr(:PyVar, ex)
        else
            ex
        end

    elseif ex isa Expr

        # x :: T
        if @capture(ex, x_ :: T_)
            pydsl_interpret_typeassert(x, T, st)

        # begin; ...; end
        elseif ex.head == :block
            args2 = [pydsl_interpret(arg, st) for arg in ex.args]
            if !isempty(args2) && ispyexpr(last(args2))
                Expr(:PyBlock, args2...)
            else
                Expr(:block, args2...)
            end

        # while cond; body; end
        elseif @capture(ex, while cond_; body_; end)
            cond, body = args
            cond2 = pydsl_interpret(cond, st)
            body2 = pydsl_interpret(body, st)
            Expr(:while, cond2, body2)

        # for v = x; body; end
        elseif @capture(ex, for v_ = x_; body_; end)
            x2 = pydsl_interpret(x, st)
            v2 = pydsl_interpret_lhs(v, st, ispyexpr(x2))
            body2 = pydsl_interpret(body, st)
            if ispyexpr(x2)
                Expr(:PyFor, Expr(:PyAssign, v2, x2), body2)
            else
                :(for $v2 = $x2; $body2; end)
            end

        # if cond; body; end
        elseif ex.head in (:if, :elseif) && length(ex.args) in (2, 3)
            cond, body = ex.args[1:2]
            ebody = length(ex.args) == 3 ? ex.args[3] : nothing
            cond2 = pydsl_interpret(cond, st)
            body2 = pydsl_interpret(body, st)
            ebody2 = pydsl_interpret(ebody, st)
            if ispyexpr(body2) || ispyexpr(ebody2)
                Expr(:PyIf, cond2, body2, ebody2)
            elseif length(ex.args) == 3
                Expr(ex.head, cond2, body2, ebody2)
            else
                Expr(ex.head, cond2, body2)
            end

        # x || y
        elseif @capture(ex, x_ || y_)
            x2 = pydsl_interpret(x, st)
            y2 = pydsl_interpret(y, st)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyOr, x2, y2)
            else
                :($x2 || y2)
            end

        # x && y
        elseif @capture(ex, x_ && y_)
            x2 = pydsl_interpret(x, st)
            y2 = pydsl_interpret(y, st)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyAnd, x2, y2)
            else
                :($x2 && $y2)
            end

        # lhs = rhs
        elseif @capture(ex, lhs_ = rhs_)
            rhs2 = pydsl_interpret(rhs, st)
            lhs2 = pydsl_interpret_lhs(lhs, st, ispyexpr(rhs2))
            if ispyexpr(rhs2)
                Expr(:PyAssign, lhs2, rhs2)
            else
                :($lhs2 = $rhs2)
            end

        # lhs += rhs, etc
        elseif haskey(PYDSL_IOPS, ex.head) && length(ex.args) == 2
            lhs, rhs = ex.args
            lhs2 = pydsl_interpret(lhs, st)
            rhs2 = pydsl_interpret(rhs, st)
            if ispyexpr(lhs2)
                Expr(:PyInPlaceAssign, ex.head, lhs2, rhs2)
            else
                Expr(ex.head, lhs2, rhs2)
            end

        # x.k
        elseif @capture(ex, x_.k_)
            x2 = pydsl_interpret(x, st)
            if k isa Symbol
                k2 = ispyexpr(x2) ? Expr(:PyObject_From, string(k)) : k
            else
                k2 = pydsl_interpret(k, st)
            end
            if ispyexpr(x2)
                Expr(:PyObject_GetAttr, x2, k2)
            else
                :($x2.$k2)
            end

        # x[ks...]
        elseif @capture(ex, x_[ks__])
            x2 = pydsl_interpret(x, st)
            ks2 = [pydsl_interpret(k, st) for k in ks]
            if ispyexpr(x2)
                if length(ks2) == 1
                    Expr(:PyObject_GetItem, x2, ks2[1])
                else
                    Expr(:PyObject_GetItem, x2, Expr(:PyTuple, ks2...))
                end
            else
                :($x2[$(ks2...)])
            end

        # x <: y
        elseif @capture(ex, x_ <: T_)
            x2 = pydsl_interpret(x, st)
            T2 = pydsl_interpret(T, st)
            if ispyexpr(x2) || ispyexpr(T2)
                Expr(:PyObject_IsSubclass, x2, T2)
            else
                :($x2 <: $T2)
            end

        # f(...)
        elseif @capture(ex, f_(args__) | f_(args__; kwargs__))
            pydsl_interpret_call(f, args, kwargs, st)

        # @pyimport ...
        elseif @capture(ex, (@pyimport (args__,)) | (@pyimport args__))
            res = Expr(:block)
            for arg in args
                if @capture(arg, mname_:attrs_)
                    mname = modname(st, mname)
                    if attrs isa Expr && attrs.head == :tuple
                        attrs = [modnamevarname(st, attr) for attr in attrs.args]
                    else
                        attrs = [modnamevarname(st, attrs)]
                    end
                    anames = [a for (a,_,_) in attrs]
                    vnames = [v for (_,v,_) in attrs]
                    for v in vnames
                        addpyvar(v, st)
                    end
                    push!(res.args, Expr(:PyAssign, Expr(:PyLHSTuple, [Expr(:PyLHSVar, v) for v in vnames]...), Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyTuple, anames...), 0)) )
                else
                    mname, vname, named = modnamevarname(st, arg)
                    addpyvar(vname, st)
                    push!(res.args, Expr(:PyAssign, Expr(:PyLHSVar, vname), Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyObject_From, PythonCall.PyLazyObject(named && '.' in mname ? ("*",) : ())), 0)))
                end
            end
            push!(res.args, nothing) # @pyimport(...) evaluates to nothing
            res

        # @py import ...
        elseif @capture(ex, @py imp_) && imp isa Expr && imp.head == :import
            res = Expr(:block)
            for arg in imp.args
                if arg isa Expr && arg.head == :(:)
                    mname = modname(st, arg.args[1])
                    attrs = [modnamevarname(st, arg) for arg in arg.args[2:end]]
                    anames = [a for (a,_,_) in attrs]
                    vnames = [v for (_,v,_) in attrs]
                    for v in vnames
                        addpyvar(v, st)
                    end
                    push!(res.args, Expr(:PyAssign, Expr(:PyLHSTuple, [Expr(:PyLHSVar, v) for v in vnames]...), Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyTuple, anames...), 0)) )
                else
                    mname, vname, named = modnamevarname(st, arg)
                    addpyvar(vname, st)
                    push!(res.args, Expr(:PyAssign, Expr(:PyLHSVar, vname), Expr(:PyImport_ImportModuleLevelObject, mname, nothing, nothing, Expr(:PyObject_From, PythonCall.PyLazyObject(named && '.' in mname ? ("*",) : ())), 0)))
                end
            end
            push!(res.args, nothing)
            res

        # @py raise (value)
        elseif @capture(ex, @py raise v_)
            v2 = pydsl_interpret(v, st)
            Expr(:PyRaise, v2)

        # @py raise (type) (value)
        elseif @capture(ex, @py raise t_Symbol v_)
            t2 = Symbol(:PyExc_, t)
            t2 in CAPI_EXCEPTIONS || pydsl_syntax_error(st, "unknown exception: $t")
            v2 = pydsl_interpret(v, st)
            Expr(:PyRaise, getproperty(CPython, t2), v2)

        # @py compile (code) (mode)
        elseif @capture(ex, @py compile code_String mode_String)
            Expr(:PyObject_From, PythonCall.PyCode(code, pydsl_srcstr(st), Symbol(mode)))

        # @py exec (code)
        elseif @capture(ex, @py exec code_String)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("exec"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :exec), Expr(:PyDict)))

        # @py exec (code) (globals)
        elseif @capture(ex, @py exec code_String globals_)
            globals2 = pydsl_interpret(globals, st)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("exec"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :exec), globals2))

        # @py exec (code) (globals) (locals)
        elseif @capture(ex, @py exec code_String globals_ locals_)
            globals2 = pydsl_interpret(globals, st)
            locals2 = pydsl_interpret(locals, st)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("exec"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :exec), globals2, locals2))

        # @py eval (code)
        elseif @capture(ex, @py eval code_String)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("eval"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :eval), Expr(:PyDict)))

        # @py eval (code) (globals)
        elseif @capture(ex, @py eval code_String globals_)
            globals2 = pydsl_interpret(globals, st)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("eval"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :eval), globals2))

        # @py eval (code) (globals) (locals)
        elseif @capture(ex, @py eval code_String globals_ locals_)
            globals2 = pydsl_interpret(globals, st)
            locals2 = pydsl_interpret(locals, st)
            Expr(:PyObject_Call, PythonCall.PyLazyBuiltinObject("eval"), Expr(:PyTuple, PythonCall.PyCode(code, pydsl_srcstr(st), :eval), globals2, locals2))

        # @py (...,)
        elseif @capture(ex, @py (args__,))
            args2 = [pydsl_interpret(arg, st) for arg in args]
            Expr(:PyTuple, args2...)

        # @py [...,]
        elseif @capture(ex, @py [args__,])
            args2 = [pydsl_interpret(arg, st) for arg in args]
            Expr(:PyList, args2...)

        # @py {...}
        elseif @capture(ex, @py {args__,})
            if isempty(args)
                Expr(:PyDict)
            elseif @capture(args[1], k_ : v_)
                args2 = []
                for arg in args
                    @capture(arg, k_ : v_) || pydsl_syntax_error(st, "invalid dict item: $arg")
                    k2 = pydsl_interpret(k, st)
                    v2 = pydsl_interpret(v, st)
                    push!(args2, :($k2 : $v2))
                end
                Expr(:PyDict, args2...)
            else
                args2 = []
                for arg in args
                    @capture(arg, k_ : v_) && pydsl_syntax_error(st, "invalid set item: $arg")
                    arg2 = pydsl_interpret(arg, st)
                    push!(args2, arg2)
                end
                Expr(:PySet, args2...)
            end

        # @py (builtin)
        elseif @capture(ex, @py arg_Symbol)
            Expr(:PyObject_From, PythonCall.PyLazyBuiltinObject(string(arg)))

        # @py (builtin)(...)
        elseif @capture(ex, @py f_Symbol(args__))
            f2 = Expr(:PyObject_From, PythonCall.PyLazyBuiltinObject(string(f)))
            pydsl_interpret_call(f2, args, nothing, st, interpret_f=false)

        # @py (builtin)(...; ...)
        elseif @capture(ex, @py f_Symbol(args__; kwargs__))
            f2 = Expr(:PyObject_From, PythonCall.PyLazyBuiltinObject(string(f)))
            pydsl_interpret_call(f2, args, kwargs, st, interpret_f=false)

        # @py ...
        elseif @capture(ex, @py args__)
            pydsl_syntax_error(st, "invalid use of @py")

        # @m(...)
        elseif ex.head == :macrocall && length(ex.args) ≥ 2
            pydsl_interpret(macroexpand(st.__module__, ex, recursive=false), st)

        else
            pydsl_syntax_error(st, "not implemented: $(ex.head)")
        end

    elseif ex isa LineNumberNode
        if pydsl_dropline(ex)
            Expr(:block)
        else
            st.__source__ = ex
        end

    else
        return ex
    end
end

"""
    @pydsl_interpret EXPR

Return the expression after applying the interpret step of the Julia-Python DSL.

This expression is not valid Julia. For that, the lower step must be applied.

See also: [`@pydsl_expand`](@ref).
"""
macro pydsl_interpret(ex)
    QuoteNode(pydsl_interpret(ex, PyDSLInterpretState(__module__=__module__)))
end

export @pydsl_interpret

struct PyExpr
    ex
    var :: Symbol
    tmp :: Bool
end

@kwdef mutable struct PyDSLLowerState
    tmpvars_unused :: Set{Symbol} = Set{Symbol}()
    tmpvars_used :: Set{Symbol} = Set{Symbol}()
    pyvars :: Set{Symbol} = Set{Symbol}()
    pyerrblocks :: Vector{Expr} = Vector{Expr}()
    lazyobjects :: Vector{Any} = Vector{Any}()
    __module__ :: Module
    __source__ :: LineNumberNode
end

pydsl_syntax_error(st::PyDSLLowerState, msg="") = throw(PyDSLSyntaxError(msg, st.__source__))

function pydsl_tmpvar(st::PyDSLLowerState)
    v = isempty(st.tmpvars_unused) ? gensym("pytmp") : pop!(st.tmpvars_unused)
    push!(st.tmpvars_used, v)
    v
end

function pydsl_discard_tmp(st::PyDSLLowerState, v::Symbol)
    @assert v in st.tmpvars_used
    delete!(st.tmpvars_used, v)
    push!(st.tmpvars_unused, v)
end

function pydsl_discard(st::PyDSLLowerState, ex::PyExpr)
    if ex.tmp
        pydsl_discard_tmp(st, ex.var)
    end
end

function pydsl_free_tmp(st::PyDSLLowerState, v::Symbol)
    pydsl_discard_tmp(st, v)
    quote
        $Py_DECREF($v)
        $v = 0
    end
end

function pydsl_free(st::PyDSLLowerState, ex::PyExpr)
    if ex.tmp
        pydsl_free_tmp(st, ex.var)
    else
        Expr(:block)
    end
end

function pydsl_steal(st::PyDSLLowerState, ex::PyExpr)
    if ex.tmp
        pydsl_discard_tmp(st, ex.var)
        quote
            $(ex.var) = 0
        end
    else
        quote
            $Py_INCREF($(ex.var))
        end
    end
end

function pydsl_errblock(st::PyDSLLowerState, ignorevars...)
    ex = Expr(:block, [:($Py_DECREF($v)) for v in st.tmpvars_used if v ∉ ignorevars]...)
    push!(st.pyerrblocks, ex)
    ex
end

function pydsl_lower_pyassign(lhs, rhs2::PyExpr, st::PyDSLLowerState)
    if lhs isa Symbol
        err = gensym("err")
        PyExpr(quote
            $(rhs2.ex)
            $err = $PyObject_Convert($(rhs2.var), $(PythonCall.PyObject))
            if $err == -1
                $(pydsl_errblock(st))
            end
            $lhs = $takeresult($(PythonCall.PyObject))
        end, rhs2.var, rhs2.tmp)
    elseif lhs isa Expr && lhs.head == :PyLHSIgnore
        length(lhs.args) == 0 || pydsl_syntax_error(st, "$ex")
        rhs2
    elseif lhs isa Expr && lhs.head == :PyLHSVar
        length(lhs.args) == 1 || pydsl_syntax_error(st, "$ex")
        v, = lhs.args
        v isa Symbol || pydsl_syntax_error(st, "$ex")
        push!(st.pyvars, v)
        PyExpr(quote
            $(rhs2.ex)
            $Py_DecRef($v)
            $v = $(rhs2.var)
            $Py_INCREF($v)
        end, rhs2.var, rhs2.tmp)
    elseif lhs isa Expr && lhs.head == :PyLHSItem
        length(lhs.args) == 2 || pydsl_syntax_error(st, "$ex")
        x, k = lhs.args
        ispyexpr(x) || pydsl_syntax_error(st, "$ex")
        x2 = pydsl_lower_inner(x, st)::PyExpr
        k2 = pydsl_lower_inner(aspyexpr(k), st)::PyExpr
        err = gensym("err")
        PyExpr(quote
            $(rhs2.ex)
            $(x2.ex)
            $(k2.ex)
            $err = $PyObject_SetItem($(x2.var), $(k2.var), $(rhs2.var))
            $(pydsl_free(st, x2))
            $(pydsl_free(st, k2))
            if $err == -1
                $(pydsl_errblock(st))
            end
        end, rhs2.var, rhs2.tmp)
    elseif lhs isa Expr && lhs.head == :PyLHSAttr
        length(lhs.args) == 2 || pydsl_syntax_error(st, "$ex")
        x, k = lhs.args
        ispyexpr(x) || pydsl_syntax_error(st, "$ex")
        x2 = pydsl_lower_inner(x, st)::PyExpr
        k2 = pydsl_lower_inner(aspyexpr(k), st)::PyExpr
        err = gensym("err")
        PyExpr(quote
            $(rhs2.ex)
            $(x2.ex)
            $(k2.ex)
            $err = $PyObject_SetAttr($(x2.var), $(k2.var), $(rhs2.var))
            $(pydsl_free(st, x2))
            $(pydsl_free(st, k2))
            if $err == -1
                $(pydsl_errblock(st))
            end
        end, rhs2.var, rhs2.tmp)
    else
        pydsl_syntax_error(st, "not implemented: assignment to $lhs")
    end
end

function pydsl_lower_pyinplaceassign(op, lhs, rhs2::PyExpr, st::PyDSLLowerState)
    if lhs isa Expr && lhs.head == :PyVar
        length(lhs.args) == 1 || pydsl_syntax_error(st, "$lhs")
        v = lhs.args[1]
        push!(st.pyvars, v)
        t = pydsl_tmpvar(st)
        PyExpr(quote
            # TODO: check v is not NULL
            $(rhs2.ex)
            $t = $op($v, $(rhs2.var))
            $(pydsl_free(st, rhs2))
            if $isnull($t)
                $(pydsl_errblock(st, t))
            end
            $Py_DECREF($v)
            $v = $t
            $Py_INCREF($v)
        end, t, true)
    elseif lhs isa Expr && lhs.head == :PyObject_GetItem
        length(lhs.args) == 2 || pydsl_syntax_error(st, "$lhs")
        x, k = lhs.args
        x2 = pydsl_lower_inner(aspyexpr(x), st)::PyExpr
        k2 = pydsl_lower_inner(aspyexpr(k), st)::PyExpr
        t1 = pydsl_tmpvar(st)
        t2 = pydsl_tmpvar(st)
        err = gensym("err")
        PyExpr(quote
            $(rhs2.ex)
            $(x2.ex)
            $(k2.ex)
            $t1 = $PyObject_GetItem($(x2.var), $(k2.var))
            if $isnull($t1)
                $(pydsl_errblock(st, t1, t2))
            end
            $t2 = $op($t1, $(rhs2.var))
            $(pydsl_free(st, rhs2))
            $(pydsl_free_tmp(st, t1))
            if $isnull($t2)
                $(pydsl_errblock(st, t2))
            end
            $err = $PyObject_SetItem($(x2.var), $(k2.var), $t2)
            $(pydsl_free(st, x2))
            $(pydsl_free(st, k2))
            if $err == -1
                $(pydsl_errblock(st))
            end
        end, t2, true)
    elseif lhs isa Expr && lhs.head == :PyObject_GetAttr
        length(lhs.args) == 2 || pydsl_syntax_error(st, "$lhs")
        x, k = lhs.args
        x2 = pydsl_lower_inner(aspyexpr(x), st)::PyExpr
        k2 = pydsl_lower_inner(aspyexpr(k), st)::PyExpr
        t1 = pydsl_tmpvar(st)
        t2 = pydsl_tmpvar(st)
        err = gensym("err")
        PyExpr(quote
            $(rhs2.ex)
            $(x2.ex)
            $(k2.ex)
            $t1 = $PyObject_GetAttr($(x2.var), $(k2.var))
            if $isnull($t1)
                $(pydsl_errblock(st, t1, t2))
            end
            $t2 = $op($t1, $(rhs2.var))
            $(pydsl_free(st, rhs2))
            $(pydsl_free_tmp(st, t1))
            if $isnull($t2)
                $(pydsl_errblock(st, t2))
            end
            $err = $PyObject_SetAttr($(x2.var), $(k2.var), $t2)
            $(pydsl_free(st, x2))
            $(pydsl_free(st, k2))
            if $err == -1
                $(pydsl_errblock(st))
            end
        end, t2, true)
    else
        pydsl_syntax_error(st, "not implemented: inplace assignment to $lhs")
    end
end

pydsl_maybelazy(x::QuoteNode) = begin
    z = pydsl_maybelazy(x.value)
    z === x.value ? x : z
end
pydsl_maybelazy(x::String) = PythonCall.PyInternedString(x)
pydsl_maybelazy(x::Expr) = x
pydsl_maybelazy(x::LineNumberNode) = x
pydsl_maybelazy(x) = ispyref(x) ? x : isimmutable(x) ? PythonCall.PyLazyObject(x) : x

function pydsl_lower_inner(ex, st::PyDSLLowerState)
    res = if ex isa Expr
        head = ex.head
        args = ex.args
        nargs = length(args)
        if head == :PyVar
            nargs == 1 || pydsl_syntax_error(st, "$ex")
            var, = args
            var isa Symbol || pydsl_syntax_error(st, "$ex")
            PyExpr(Expr(:block), var, false)
        elseif head == :PyObject_From
            nargs == 1 || pydsl_syntax_error(st, "$ex")
            x, = args
            x = pydsl_maybelazy(x)
            if x isa Bool
                t = gensym("pytmp")
                PyExpr(quote
                    $t = $(x ? Py_True : Py_False)()
                end, t, false)
            elseif x isa Nothing || x === :nothing
                t = gensym("pytmp")
                PyExpr(quote
                    $t = $Py_None()
                end, t, false)
            elseif x isa PythonCall.PyLazyObject ||
                    x isa PythonCall.PyInternedString ||
                    x isa PythonCall.PyLazyBuiltinObject ||
                    x isa PythonCall.PyCode
                # putting these into st.lazyobjects means we can assume the object's pointer is valid by the time we execute this code
                push!(st.lazyobjects, x)
                t = gensym("pytmp")
                PyExpr(quote
                    $t = getfield($x, :ptr)
                end, t, false)
            else
                x2 = pydsl_lower_inner(x, st)
                if x2 isa PyExpr
                    x2
                else
                    t = pydsl_tmpvar(st)
                    PyExpr(quote
                        $t = $PyObject_From($x2) :: $PyPtr
                        if $isnull($t)
                            $(pydsl_errblock(st, t))
                        end
                    end, t, true)
                end
            end
        elseif head == :PyExternObject
            nargs == 3 || pydsl_syntax_error(st, "$ex")
            x, borrow, check = args
            x2 = pydsl_lower_inner(x, st)
            ispyexpr(x2) && pydsl_syntax_error(st, "$ex")
            t = borrow ? gensym("pytmp") : pydsl_tmpvar(st)
            if check
                PyExpr(quote
                    $t = $x2 :: $PyPtr
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                end, t, !borrow)
            else
                PyExpr(quote
                    $t = $x2 :: $PyPtr
                end, t, !borrow)
            end
        elseif head == :PyObject_Convert
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            x, T = args
            x2 = pydsl_lower_inner(x, st)
            T2 = pydsl_lower_inner(T, st)
            T2 isa PyExpr && pydsl_syntax_error(st, "$ex")
            x2 isa PyExpr || pydsl_syntax_error(st, "$ex")
            err = gensym("err")
            if T2 isa Expr
                Tv = gensym("T")
                quote
                    $Tv = $T2
                    $(x2.ex)
                    $err = $PyObject_Convert($(x2.var), $T2)
                    $(pydsl_free(st, x2))
                    if $err == -1
                        $(pydsl_errblock(st))
                    end
                    $(takeresult)($Tv)
                end
            else
                quote
                    $(x2.ex)
                    $err = $PyObject_Convert($(x2.var), $T2)
                    $(pydsl_free(st, x2))
                    if $err == -1
                        $(pydsl_errblock(st))
                    end
                    $(takeresult)($T2)
                end
            end
        elseif head == :PyPtr
            nargs == 1 || pydsl_syntax_error(st, "$ex")
            x, = args
            x2 = pydsl_lower_inner(x, st)
            x2 isa PyExpr || pydsl_syntax_error(st, "$ex")
            t = gensym("pyptr")
            quote
                $(x2.ex)
                $t = $(x2.var)
                $(pydsl_steal(st, x2))
                $t
            end
        elseif head == :PyObject_Call
            if nargs == 1
                f, = args
                f2 = pydsl_lower_inner(aspyexpr(f), st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(f2.ex)
                    $t = $PyObject_CallObject($(f2.var), $PyNULL)
                    $(pydsl_free(st, f2))
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            elseif nargs == 2
                f, varargs = args
                f2 = pydsl_lower_inner(aspyexpr(f), st) :: PyExpr
                varargs isa Expr && varargs.head == :PyTuple || pydsl_syntax_error(st, "$ex")
                varargs2 = pydsl_lower_inner(varargs, st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(f2.ex)
                    $(varargs2.ex)
                    $t = $PyObject_CallObject($(f2.var), $(varargs2.var))
                    $(pydsl_free(st, f2))
                    $(pydsl_free(st, varargs2))
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            elseif nargs == 3
                f, varargs, kwargs = args
                f2 = pydsl_lower_inner(aspyexpr(f), st) :: PyExpr
                varargs isa Expr && varargs.head == :PyTuple || pydsl_syntax_error(st, "$ex")
                varargs2 = pydsl_lower_inner(varargs, st) :: PyExpr
                kwargs isa Expr && kwargs.head == :PyKwargs || pydsl_syntax_error(st, "$ex")
                kwargs2 = pydsl_lower_inner(kwargs, st) :: PyExpr
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $(f2.ex)
                    $(varargs2.ex)
                    $(kwargs2.ex)
                    $t = $PyObject_CallObject($(f2.var), $(varargs2.var), $(kwargs2.var))
                    $(pydsl_free(st, f2))
                    $(pydsl_free(st, varargs2))
                    $(pydsl_free(st, kwargs2))
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            else
                pydsl_syntax_error(st, "$ex")
            end
        elseif head == :PyTuple
            if any(ex -> ex isa Expr && ex.head == :..., args)
                pydsl_syntax_error(st, "splatting into a tuple not implemented")
            else
                t = pydsl_tmpvar(st)
                err = gensym("err")
                PyExpr(quote
                    $t = $PyTuple_New($(length(args)))
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                    $([
                        begin
                            arg2 = pydsl_lower_inner(aspyexpr(arg), st) :: PyExpr
                            quote
                                $(arg2.ex)
                                $err = $PyTuple_SetItem($t, $(i-1), $(arg2.var))
                                $(pydsl_steal(st, arg2))
                                if $err == -1
                                    $(pydsl_errblock(st))
                                end
                            end
                        end
                        for (i,arg) in enumerate(args)
                    ]...)
                end, t, true)
            end
        elseif head == :PyList
            if any(ex -> ex isa Expr && ex.head == :..., args)
                pydsl_syntax_error(st, "splatting into a list not implemented")
            else
                t = pydsl_tmpvar(st)
                err = gensym("err")
                PyExpr(quote
                    $t = $PyList_New($(length(args)))
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                    $([
                        begin
                            arg2 = pydsl_lower_inner(aspyexpr(arg), st) :: PyExpr
                            quote
                                $(arg2.ex)
                                $err = $PyList_SetItem($t, $(i-1), $(arg2.var))
                                $(pydsl_steal(st, arg2))
                                if $err == -1
                                    $(pydsl_errblock(st))
                                end
                            end
                        end
                        for (i,arg) in enumerate(args)
                    ]...)
                end, t, true)
            end
        elseif head == :PySet
            if any(ex -> ex isa Expr && ex.head == :..., args)
                pydsl_syntax_error(st, "splatting into a set not implemented")
            else
                t = pydsl_tmpvar(st)
                err = gensym("err")
                PyExpr(quote
                    $t = $PySet_New($C_NULL)
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                    $([
                        begin
                            arg2 = pydsl_lower_inner(aspyexpr(arg), st) :: PyExpr
                            quote
                                $(arg2.ex)
                                $err = $PySet_Add($t, $(arg2.var))
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
        elseif head == :PyDict
            if any(ex -> ex isa Expr && ex.head == :..., args)
                pydsl_syntax_error(st, "splatting into a dict not implemented")
            else
                t = pydsl_tmpvar(st)
                err = gensym("err")
                PyExpr(quote
                    $t = $PyDict_New()
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                    $([
                        begin
                            @capture(arg, k_ : v_) || pydsl_syntax_error(st, "invalid dict item: $arg")
                            k2 = pydsl_lower_inner(aspyexpr(k), st) :: PyExpr
                            v2 = pydsl_lower_inner(aspyexpr(v), st) :: PyExpr
                            quote
                                $(k2.ex)
                                $(v2.ex)
                                $err = $PyDict_SetItem($t, $(k2.var), $(v2.var))
                                $(pydsl_free(st, k2))
                                $(pydsl_free(st, v2))
                                if $err == -1
                                    $(pydsl_errblock(st))
                                end
                            end
                        end
                        for (i,arg) in enumerate(args)
                    ]...)
                end, t, true)
            end
        elseif head == :PyTypeAssert
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            x, T = args
            ispyexpr(x) || pydsl_syntax_error(st, "$ex")
            ispyexpr(x) || pydsl_syntax_error(st, "$ex")
            x2 = pydsl_lower_inner(x, st)::PyExpr
            T2 = pydsl_lower_inner(T, st)::PyExpr
            ans = gensym("ans")
            PyExpr(quote
                $(x2.ex)
                $(T2.ex)
                $ans = $PyObject_IsInstance($(x2.var), $(T2.var))
                $(pydsl_free(st, T2))
                if $ans == -1
                    $(pydsl_errblock(st))
                elseif $ans == 0
                    $PyErr_SetString($PyExc_TypeError(), "in type assert, got a '$($PyType_Name($Py_Type($(x2.var))))'")
                    $(pydsl_errblock(st))
                end
            end, x2.var, x2.tmp)
        elseif head == :PyImport_ImportModuleLevelObject
            nargs == 5 || pydsl_syntax_error(st, "$ex")
            mod, gls, lcs, frs, lvl = args
            lvl isa Int || pydsl_syntax_error(st, "$ex")
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
                $t = $PyImport_ImportModuleLevelObject($(mod2.var), $(gls2.var), $(lcs2.var), $(frs2.var), $lvl)
                $(pydsl_free(st, mod2))
                $(pydsl_free(st, gls2))
                $(pydsl_free(st, lcs2))
                $(pydsl_free(st, frs2))
                if $isnull($t)
                    $(pydsl_errblock(st, t))
                end
            end, t, true)
        elseif head == :PyIgnore
            nargs == 1 || pydsl_syntax_error(st, "$ex")
            inner, = args
            inner2 = pydsl_lower_inner(inner, st)
            if inner2 isa PyExpr
                quote
                    $(inner2.ex)
                    $(pydsl_free(st, inner2))
                    nothing
                end
            else
                inner2
            end
        elseif head == :PyBlock
            args2 = [pydsl_lower_inner(i==nargs ? aspyexpr(arg) : ignorepyexpr(arg), st) for (i,arg) in enumerate(args)]
            isempty(args2) && push!(args2, pydsl_lower_inner(aspyexpr(nothing), st))
            PyExpr(Expr(:block, [arg for arg in args2[1:end-1]]..., last(args2).ex), last(args2).var, last(args2).tmp)
        elseif head == :PyAssign
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            lhs, rhs = args
            rhs2 = pydsl_lower_inner(aspyexpr(rhs), st)::PyExpr
            pydsl_lower_pyassign(lhs, rhs2, st)
        elseif head == :PyInPlaceAssign
            nargs == 3 || pydsl_syntax_error(st, "$ex")
            op, lhs, rhs = args
            rhs2 = pydsl_lower_inner(aspyexpr(rhs), st)::PyExpr
            pydsl_lower_pyinplaceassign(getproperty(CPython, PYDSL_IOPS[op]), lhs, rhs2, st)
        elseif head == :PyTruth
            nargs == 1 || pydsl_syntax_error(st, "$ex")
            inner, = args
            inner2 = pydsl_lower_inner(inner, st)
            if inner2 isa PyExpr
                t = gensym("truth")
                quote
                    $(inner2.ex)
                    $t = $PyObject_IsTrue($(inner2.var))
                    $(pydsl_free(st, inner2))
                    if $t == -1
                        $(pydsl_errblock(st))
                    end
                    $t != 0
                end
            else
                inner2
            end
        elseif head == :PyIf
            nargs == 3 || pydsl_syntax_error(st, "$ex")
            cond, body, ebody = args
            cond2 = pydsl_lower_inner(truthpyexpr(cond), st)
            tmp = gensym("ifans")
            body2 = pydsl_lower_inner(aspyexpr(body), st)::PyExpr
            body2 = quote
                $(body2.ex)
                $tmp = $(body2.var)
                $(pydsl_steal(st, body2))
            end
            ebody2 = pydsl_lower_inner(aspyexpr(ebody), st)::PyExpr
            ebody2 = quote
                $(ebody2.ex)
                $tmp = $(ebody2.var)
                $(pydsl_steal(st, ebody2))
            end
            t = pydsl_tmpvar(st)
            c = gensym("cond")
            PyExpr(quote
                $c = $cond2
                if $c
                    $body2
                else
                    $ebody2
                end
                $t = $tmp
            end, t, true)
        elseif head == :PyAnd
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            x, y = args
            x2 = pydsl_lower_inner(aspyexpr(x), st)::PyExpr
            tmp = gensym("lhs")
            cond = gensym("cond")
            x2 = quote
                $(x2.ex)
                $tmp = $(x2.var)
                $cond = $PyObject_IsTrue($tmp)
                if $cond == -1
                    $(pydsl_errblock(st))
                end
                $(pydsl_steal(st, x2))
            end
            y2 = pydsl_lower_inner(aspyexpr(y), st)::PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(x2.args...)
                if $cond != 0
                    $(y2.ex)
                    $Py_DECREF($tmp)
                    $tmp = $(y2.var)
                    $(pydsl_steal(st, y2))
                end
                $t = $tmp
            end, t, true)
        elseif head == :PyOr
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            x, y = args
            x2 = pydsl_lower_inner(aspyexpr(x), st)::PyExpr
            tmp = gensym("lhs")
            cond = gensym("cond")
            x2 = quote
                $(x2.ex)
                $tmp = $(x2.var)
                $cond = $PyObject_IsTrue($tmp)
                if $cond == -1
                    $(pydsl_errblock(st))
                end
                $(pydsl_steal(st, x2))
            end
            y2 = pydsl_lower_inner(aspyexpr(y), st)::PyExpr
            t = pydsl_tmpvar(st)
            PyExpr(quote
                $(x2.args...)
                if $cond == 0
                    $(y2.ex)
                    $Py_DECREF($tmp)
                    $tmp = $(y2.var)
                    $(pydsl_steal(st, y2))
                end
                $t = $tmp
            end, t, true)
        elseif head in (:PyObject_Is, :PyObject_IsNot)
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            x, y = args
            x2 = pydsl_lower_inner(aspyexpr(x), st)::PyExpr
            y2 = pydsl_lower_inner(aspyexpr(y), st)::PyExpr
            op =
                head == :PyObject_Is    ? (==) :
                head == :PyObject_IsNot ? (!=) :
                @assert false
            t = gensym("ans")
            quote
                $(x2.ex)
                $(y2.ex)
                $t = $(op)($(x2.var), $(y2.var))
                $(pydsl_free(st, x2))
                $(pydsl_free(st, y2))
                $t
            end
        elseif haskey(PYDSL_OPS, head)
            opnargs, _, rettype = PYDSL_OPS[head]
            nargs in opnargs || pydsl_syntax_error(st, "$ex")
            args2 = [pydsl_lower_inner(aspyexpr(arg), st)::PyExpr for arg in args]
            if rettype == :py
                t = pydsl_tmpvar(st)
                PyExpr(quote
                    $([arg2.ex for arg2 in args2]...)
                    $t = $(getproperty(CPython, head))($([arg2.var for arg2 in args2]...))
                    $([pydsl_free(st, arg2) for arg2 in args2]...)
                    if $isnull($t)
                        $(pydsl_errblock(st, t))
                    end
                end, t, true)
            elseif rettype == :bool
                t = gensym("bool")
                quote
                    $([arg2.ex for arg2 in args2]...)
                    $t = $(getproperty(CPython, head))($([arg2.var for arg2 in args2]...))
                    $([pydsl_free(st, arg2) for arg2 in args2]...)
                    if $t == -1
                        $(pydsl_errblock(st, t))
                    end
                    $t != 0
                end
            elseif rettype == :int
                t = gensym("int")
                quote
                    $([arg2.ex for arg2 in args2]...)
                    $t = $(getproperty(CPython, head))($([arg2.var for arg2 in args2]...))
                    $([pydsl_free(st, arg2) for arg2 in args2]...)
                    if $t == -1
                        $(pydsl_errblock(st, t))
                    end
                    $t
                end
            elseif rettype == :void
                t = gensym("err")
                quote
                    $([arg2.ex for arg2 in args2]...)
                    $t = $(getproperty(CPython, head))($([arg2.var for arg2 in args2]...))
                    $([pydsl_free(st, arg2) for arg2 in args2]...)
                    if $t == -1
                        $(pydsl_errblock(st, t))
                    end
                    nothing
                end
            else
                @assert false
            end
        elseif head == :PyFor
            nargs == 2 || pydsl_syntax_error(st, "$ex")
            assign, body = args
            assign isa Expr && assign.head == :PyAssign && length(assign.args) == 2 || pydsl_syntax_error(st, "$ex")
            v, x = assign.args
            x2 = pydsl_lower_inner(x, st)
            it = pydsl_tmpvar(st)
            freex2 = pydsl_free(st, x2)
            iterr = pydsl_errblock(st, it)
            tmp = pydsl_tmpvar(st)
            getnexterr = pydsl_errblock(st, tmp)
            getnext = PyExpr(quote
                $tmp = $PyIter_Next($it)
                if $isnull($tmp)
                    if $PyErr_IsSet()
                        $getnexterr
                    else
                        break
                    end
                end
            end, tmp, true)
            assignnext = pydsl_lower_pyassign(v, getnext, st)
            freetmp = pydsl_free(st, assignnext)
            body2 = pydsl_lower_inner(ignorepyexpr(body), st)
            quote
                $(x2.ex)
                $it = $PyObject_GetIter($(x2.var))
                $freex2
                if $isnull($it)
                    $iterr
                end
                while true
                    $(assignnext.ex)
                    $freetmp
                    $body2
                end
                $(pydsl_free_tmp(st, it))
                nothing
            end
        elseif head == :PyRaise
            if nargs == 1
                v, = args
                v2 = pydsl_lower_inner(aspyexpr(v), st)::PyExpr
                quote
                    $(v2.ex)
                    if $Py_TypeCheckFast($(v2.var), $Py_TPFLAGS_BASE_EXC_SUBCLASS)
                        $PyErr_SetObject($Py_Type($(v2.var)), $(v2.var))
                    elseif $Py_TypeCheckFast($(v2.var), $Py_TPFLAGS_TYPE_SUBCLASS) && $PyType_IsSubtypeFast($(v2.var), $Py_TPFLAGS_BASE_EXC_SUBCLASS)
                        $PyErr_SetNone($(v2.var))
                    else
                        $PyErr_SetString($PyExc_TypeError(), "exceptions must derive from BaseException")
                    end
                    $(pydsl_free(st, v2))
                    $(pydsl_errblock(st))
                end
            elseif nargs == 2
                t, v = args
                v2 = pydsl_lower_inner(aspyexpr(v), st)::PyExpr
                quote
                    $(v2.ex)
                    $PyErr_SetObject($t(), $(v2.var))
                    $(pydsl_free(st, v2))
                    $(pydsl_errblock(st))
                end
            else
                pydsl_syntax_error(st, "$ex")
            end
        elseif head == :block
            args2 = [pydsl_lower_inner(i==nargs ? nopyexpr(arg) : ignorepyexpr(arg), st) for (i,arg) in enumerate(args)]
            Expr(:block, args2...)
        elseif head == :call
            args2 = [pydsl_lower_inner(nopyexpr(arg), st) for arg in args]
            Expr(:call, args2...)
        elseif head == :while && nargs == 2
            cond, body = args
            cond2 = pydsl_lower_inner(truthpyexpr(cond), st)
            body2 = pydsl_lower_inner(ignorepyexpr(body), st)
            Expr(:while, cond2, body2)
        elseif head in (:if, :elseif) && nargs in (2,3)
            cond, rest... = args
            cond2 = pydsl_lower_inner(truthpyexpr(cond), st)
            rest2 = [pydsl_lower_inner(nopyexpr(x), st) for x in rest]
            Expr(head, cond2, rest2...)
        elseif head == :. && nargs == 2
            x, k = args
            x2 = pydsl_lower_inner(nopyexpr(x), st)
            k2 = pydsl_lower_inner(nopyexpr(k), st)
            Expr(:., x2, k2)
        else
            pydsl_syntax_error(st, "not implemented: Expr($(repr(head)), ...)")
        end
    elseif ex isa LineNumberNode
        if pydsl_dropline(ex)
            Expr(:block)
        else
            st.__source__ = ex
        end
    else
        ex
    end
    @assert ispyexpr(ex) == isa(res, PyExpr)
    res
end

function pydsl_lower(ex, st::PyDSLLowerState; onpyerror, onjlerror)
    src = st.__source__
    ex = nopyexpr(ex)
    ex = pydsl_lower_inner(ex, st)
    @assert isempty(st.tmpvars_used)
    @assert !ispyexpr(ex)
    inits = Any[:($v :: $PyPtr = 0) for v in union(st.pyvars, st.tmpvars_unused)]
    decrefs = Any[:($Py_DecRef($v)) for v in st.pyvars]
    tmpdecrefs = Any[:($Py_DecRef($v)) for v in st.tmpvars_unused]
    if !isempty(st.lazyobjects)
        # ensure lazy objects have non-null pointers
        lazyinited = Ref(false)
        nullex = mapreduce(o->:($isnull($pyptr($o))), (x,y)->:($x || $y), st.lazyobjects)
        push!(inits, quote
            if !$lazyinited[]
                if $nullex
                    $(pydsl_errblock(st))
                end
                $lazyinited[] = true
            end
        end)
        # keep a reference to the lazy objects
        push!(decrefs, lazyinited)
    end
    for block in st.pyerrblocks
        push!(block.args, onpyerror)
    end
    if onjlerror === :impossible
        ans = gensym("ans")
        quote
            let
                $src
                $(inits...)
                $ans = $(ex)
                $(decrefs...)
                $ans
            end
        end
    else
        ispyerr = gensym("ispyerr")
        for block in st.pyerrblocks
            pushfirst!(block.args, :($ispyerr = true))
        end
        quote
            let
                $src
                $ispyerr = false
                $(inits...)
                try
                    $(ex)
                catch exc
                    if $ispyerr
                        rethrow()
                    else
                        $(tmpdecrefs...)
                        $(onjlerror)
                    end
                finally
                    $(decrefs...)
                end
            end
        end
    end
end

function pydsl_macro(__module__, __source__, ex, kwargs...)
    onpyerror = :($(PythonCall.pythrow)())
    onjlerror = :($rethrow())
    interpret = true
    for kwarg in kwargs
        kwarg isa Expr && kwarg.head == :(=) && length(kwarg.args) == 2 || error("invalid keyword argument: $(kwarg)")
        k, v = kwarg.args
        if k == :onpyerror
            onpyerror = v
        elseif k == :onjlerror
            onjlerror = v
        elseif k == :interpret
            v isa Bool || error("interpret argument must be a Bool")
            interpret = v
        else
            error("invalid keyword argument: $k")
        end
    end
    # interpret
    if interpret
        ex = pydsl_interpret(ex, PyDSLInterpretState(__module__=__module__, __source__=__source__))
    end
    # lower
    ex = pydsl_lower(ex, PyDSLLowerState(__module__=__module__, __source__=__source__); onpyerror=onpyerror, onjlerror=onjlerror)
    # strip out LineNumberNodes from this file
    ex = MacroTools.prewalk(ex -> ex isa LineNumberNode && pydsl_dropline(ex) ? Expr(:block) : ex, ex)
end

"""
    @pydsl EXPR [onpyerror=pythrow()] [onjlerror=rethrow()] [interpret=true]

Execute the given expression, interpreting it in the Julia-Python DSL.

# Keyword Arguments

- `onpyerror`: The action to take when a Python error occurs. It must `throw` or `return`.
- `onjlerror`: The action to take when a Julia error occurs. It must `throw` or `return`. The special value `impossible` indicates that the Julia errors cannot happen, so the resulting expression is not wrapped in a `try`/`catch` block. In this case, any Python variables are decref'd at the end, so you MUST NOT use this when the expression contains Python variables and might not execute to the end (e.g. if it contains a `throw` or `return`).
- `interpret`: When false, do not apply the interpret step of the language.
"""
macro pydsl(ex, kwargs...)
    esc(pydsl_macro(__module__, __source__, ex, kwargs...))
end

export @pydsl

"""
    @pydsl_nojlerror EXPR ...

Equivalent to `@pydsl EXPR onjlerror=impossible ...`.
"""
macro pydsl_nojlerror(ex, kwargs...)
    esc(pydsl_macro(__module__, __source__, ex, Expr(:(=), :onjlerror, :impossible), kwargs...))
end

export @pydsl_nojlerror

"""
    @pydsl_expand EXPR ...

Similar to `@macroexpand @pydsl EXPR ...` but makes the output much more readable.

See also: [`@pydsl_interpret`](@ref).
"""
macro pydsl_expand(args...)
    ex = pydsl_macro(__module__, __source__, args...)
    # ex = MacroTools.striplines(ex)
    ex = MacroTools.flatten(ex)
    ex = MacroTools.unresolve(ex)
    ex = MacroTools.resyntax(ex)
    ex = MacroTools.gensym_ids(ex)
    QuoteNode(ex)
end

export @pydsl_expand
