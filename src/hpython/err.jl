iserr(ctx::Context) = ctx._c.PyErr_Occurred() != C.PyNULL

errclear(ctx::Context) = ctx._c.PyErr_Clear()
(b::Builtin{:errclear})() = errclear(b.ctx)

function errset(ctx::Context, t)
    @autohdl ctx t
    ctx._c.PyErr_SetNone(ctx.cptr(t))
    @autoclosehdl ctx t
end
function errset(ctx::Context, t, v)
    @autohdl ctx t v
    ctx._c.PyErr_SetObject(ctx.cptr(t), ctx.cptr(v))
    @autoclosehdl ctx t v
end
function errset(ctx::Context, t, v::String)
    @autohdl ctx t
    ctx._c.PyErr_SetString(ctx.cptr(t), v)
    @autoclosehdl ctx t
end
(b::Builtin{:errset})(args...) = errset(b.ctx, args...)

function errget(ctx::Context, normalize::Bool=false)
    t = Ref(C.PyNULL)
    v = Ref(C.PyNULL)
    b = Ref(C.PyNULL)
    ctx._c.PyErr_Fetch(t, v, b)
    if normalize
        ctx._c.PyErr_NormalizeException(t, v, b)
    end
    (ctx.newhdl(t[]), ctx.newhdl(v[]), ctx.newhdl(b[]))
end
(b::Builtin{:errget})(args...) = errget(b.ctx, args...)
