pyobject(ctx::Context, x::PyAnyHdl) = ctx.duphdl(x)
pyobject(ctx::Context, x::Builtin) = ctx.duphdl(ctx.autohdl(x))
pyobject(ctx::Context, x::Nothing) = pyobject(ctx, ctx.None)
pyobject(ctx::Context, x::Bool) = x ? pyobject(ctx, ctx.True) : pyobject(ctx, ctx.False)
pyobject(ctx::Context, x::Union{String,SubString{String}}) = ctx.str(x)
(ctx::Context)(x) = pyobject(ctx, x)

function pyhasattr(ctx::Context, x, k)
    ans = BoolOrErr()
    @autohdl ctx x k
    ans = BoolOrErr(ctx._c.PyObject_HasAttr(ctx.cptr(x), ctx.cptr(k)))
    @autoclosehdl ctx x k
    return ans
end
function pyhasattr(ctx::Context, x::PyHdl, k::PyHdl)
    # this special case can return Bool instead of BoolOrErr
    ctx._c.PyObject_HasAttr(ctx.cptr(x), ctx.cptr(k)) != 0
end
(f::Builtin{:hasattr})(x, k) = pyhasattr(f.ctx, x, k)

function pygetattr(ctx::Context, x, k)
    ans = PyNULL
    @autohdl ctx x k
    ans = ctx.newhdl(ctx._c.PyObject_GetAttr(ctx.cptr(x), ctx.cptr(k)))
    @autoclosehdl ctx x k
    return ans
end
(f::Builtin{:getattr})(x, k) = pygetattr(f.ctx, x, k)

function pysetattr(ctx::Context, x, k, v)
    ans = VoidOrErr()
    @autohdl ctx x k v
    ans = VoidOrErr(ctx._c.PyObject_SetAttr(ctx.cptr(x), ctx.cptr(k), ctx.cptr(v)))
    @autoclosehdl ctx x k v
    return ans
end
(f::Builtin{:setattr})(x, k, v) = pysetattr(f.ctx, x, k, v)

function pyascii(ctx::Context, x)
    ans = PyNULL
    @autohdl ctx x
    ans = ctx.newhdl(ctx._c.PyObject_ASCII(ctx.cptr(x)))
    @autoclosehdl ctx x
    return ans
end
pyascii(::Type{T}, ctx::Context, x) where {T} = pystr_convert(T, ctx, pyascii(ctx, x).auto)
(f::Builtin{:ascii})(x) = pyascii(f.ctx, x)
(f::Builtin{:ascii})(::Type{T}, x) where {T} = pyascii(T, f.ctx, x)

function pyrepr(ctx::Context, x)
    ans = PyNULL
    @autohdl ctx x
    ans = ctx.newhdl(ctx._c.PyObject_Repr(ctx.cptr(x)))
    @autoclosehdl ctx x
    return ans
end
pyrepr(::Type{T}, ctx::Context, x) where {T} = pystr_convert(T, ctx, pyrepr(ctx, x).auto)
(f::Builtin{:repr})(x) = pyrepr(f.ctx, x)
(f::Builtin{:repr})(::Type{T}, x) where {T} = pyrepr(T, f.ctx, x)
