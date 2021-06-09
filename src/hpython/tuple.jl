pytuple(ctx::Context) = ctx.newhdl(ctx._c.PyTuple_New(0))
pytuple(ctx::Context, xs::PyAnyHdl) = ctx.call(ctx.tuple, xs)
function pytuple(ctx::Context, xs)
    ans = ctx.newhdl(ctx._c.PyTuple_New(length(xs)))
    ctx.iserr(ans) && return PyNULL
    for (i,x) in enumerate(xs)
        xh = ctx(x)
        if ctx.iserr(xh)
            ctx.closehdl(ans)
            return PyNULL
        end
        err = ctx._c.PyTuple_SetItem(ctx.cptr(ans), i-1, ctx.stealcptr(xh))
        if err == -1
            ctx.closehdl(ans)
            return PyNULL
        end
    end
    return ans
end
(b::Builtin{:tuple})(args...) = pytuple(b.ctx, args...)
