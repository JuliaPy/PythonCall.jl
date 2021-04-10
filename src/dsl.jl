using MacroTools

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
])

ispyexpr(ex) = ex isa Expr && ex.head in PYEXPR_HEADS

pydsl_interpret(ex, ispyvar) = begin
    if ex isa Symbol
        if get!(ispyvar, ex, false)
            Expr(:PyVar, ex)
        else
            ex
        end
    elseif ex isa Expr
        # x::T
        if @capture(ex, (xs__,)::Py)
            xs2 = [pydsl_interpret(x, ispyvar) for x in xs]
            Expr(:PyTuple, xs2...)
        elseif @capture(ex, lhs_::Py)
            lhs2 = pydsl_interpret(lhs, ispyvar)
            if ispyexpr(lhs2)
                lhs2
            else
                Expr(:PyObject_From, lhs2)
            end
        elseif @capture(ex, lhs_::T_)
            lhs2 = pydsl_interpret(lhs, ispyvar)
            T2 = pydsl_interpret(T, ispyvar)
            if ispyexpr(lhs2)
                Expr(:PyObject_Convert, lhs2, T2)
            else
                Expr(:(::), lhs2, T2)
            end

        # x.k
        elseif ex.head == :. && length(ex.args) == 2
            # NOTE: @capture(ex, x_.y_) unpacks y when it was as QuoteNode
            x, y = ex.args
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2)
                Expr(:PyObject_GetAttr, x2, y2)
            else
                Expr(:., x2, y2)
            end

        # x[k...]
        elseif @capture(ex, x_[ks__])
            x2 = pydsl_interpret(x, ispyvar)
            ks2 = [pydsl_interpret(k, ispyvar) for k in ks]
            if ispyexpr(x2)
                if length(ks2) == 1
                    Expr(:PyObject_GetItem, x2, ks2[1])
                else
                    Expr(:PyObject_GetItem, x2, Expr(:PyTuple, ks2...))
                end
            else
                Expr(:ref, x2, ks2...)
            end

        # length
        elseif @capture(ex, length(x_))
            x2 = pydsl_interpret(x, ispyvar)
            if ispyexpr(x2)
                Expr(:PyObject_Length, x2)
            else
                Expr(:call, :length, x2)
            end

        # arithmetic
        elseif @capture(ex, +(args__))
            args2 = [pydsl_interpret(arg, ispyvar) for arg in args]
            if any(ispyexpr, args2)
                if length(args2) == 1
                    Expr(:PyNumber_Positive, args2[1])
                else
                    ret = args2[1]
                    for arg in args2[2:end]
                        ret = Expr(:PyNumber_Add, ret, arg)
                    end
                    ret
                end
            else
                Expr(:call, :+, args2...)
            end
        elseif @capture(ex, -x_)
            x2 = pydsl_interpret(x, ispyvar)
            if ispyexpr(x2)
                Expr(:PyNumber_Negative, x2)
            else
                Expr(:call, :-, x2)
            end
        elseif @capture(ex, x_-y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Subtract, x2, y2)
            else
                Expr(:call, :-, x2, y2)
            end
        elseif @capture(ex, *(args__))
            args2 = [pydsl_interpret(arg, ispyvar) for arg in args]
            if any(ispyexpr, args2)
                ret = args2[1]
                for arg in args2[2:end]
                    ret = Expr(:PyNumber_Multiply, ret, arg)
                end
                ret
            else
                Expr(:call, :*, args2...)
            end
        elseif @capture(ex, x_/y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_TrueDivide, x2, y2)
            else
                Expr(:call, :/, x2, y2)
            end
        elseif @capture(ex, x_÷y_) || @capture(ex, div(x_, y_))
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_FloorDivide, x2, y2)
            else
                Expr(:call, :÷, x2, y2)
            end
        elseif @capture(ex, x_%y_) || @capture(ex, mod(x_, y_))
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Remainder, x2, y2)
            else
                Expr(:call, :%, x2, y2)
            end
        elseif @capture(ex, abs(x_))
            x2 = pydsl_interpret(x, ispyvar)
            if ispyexpr(x2)
                Expr(:PyNumber_Absolute, x2)
            else
                Expr(:call, :abs, x2)
            end
        elseif @capture(ex, ~x_)
            x2 = pydsl_interpret(x, ispyvar)
            if ispyexpr(x2)
                Expr(:PyNumber_Invert, x2)
            else
                Expr(:call, :~, x2)
            end
        elseif @capture(ex, x_^y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Power, x2, y2)
            else
                Expr(:call, :^, x2, y2)
            end
        elseif @capture(ex, powermod(x_, y_, z_))
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            z2 = pydsl_interpret(z, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2) || ispyexpr(z2)
                Expr(:PyNumber_Power, x2, y2, z2)
            else
                Expr(:call, :powermod, x2, y2, z2)
            end
        elseif @capture(ex, x_<<y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Lshift, x2, y2)
            else
                Expr(:call, :(<<), x2, y2)
            end
        elseif @capture(ex, x_>>y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Rshift, x2, y2)
            else
                Expr(:call, :(>>), x2, y2)
            end
        elseif @capture(ex, x_&y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_And, x2, y2)
            else
                Expr(:call, :&, x2, y2)
            end
        elseif @capture(ex, x_|y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Or, x2, y2)
            else
                Expr(:call, :|, x2, y2)
            end
        elseif @capture(ex, x_⊻y_)
            x2 = pydsl_interpret(x, ispyvar)
            y2 = pydsl_interpret(y, ispyvar)
            if ispyexpr(x2) || ispyexpr(y2)
                Expr(:PyNumber_Xor, x2, y2)
            else
                Expr(:call, :⊻, x2, y2)
            end

        # other function calls
        elseif ex.head == :call && length(ex.args) > 0
            args2 = [pydsl_interpret(arg, ispyvar) for arg in ex.args]
            if ispyexpr(args2[1])
                Expr(:PyObject_Call, args2...)
            else
                Expr(:call, args2...)
            end
        elseif ex.head == :kw && length(ex.args) == 2
            k, v = ex.args
            v2 = pydsl_interpret(v, ispyvar)
            Expr(:kw, k, v2)
        elseif ex.head == :parameters
            args = [pydsl_interpret(arg, ispyvar) for arg in ex.args]
            Expr(:parameters, args...)

        else
            error("not implemented: $ex")
        end
    else
        return ex
    end
end

macro pydsl_interpret(ex)
    QuoteNode(pydsl_interpret(ex, Dict{Symbol, Bool}()))
end

export @pydsl_interpret
