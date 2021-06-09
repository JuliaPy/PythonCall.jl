function pyimport(ctx::Context, m)
    ans = PyNULL
    @autohdl ctx m
    ans = ctx.newhdl(ctx._c.PyImport_Import(ctx.cptr(m)))
    @autoclosehdl ctx m
    return ans
end
(f::Builtin{:import})(m) = pyimport(f.ctx, m)

pycall(ctx::Context, f, args...; kwargs...) =
    if !isempty(kwargs)
        error("not implemented: keyword arguments")
    elseif !isempty(args)
        pycallargs(ctx, f, ctx.tuple(args).auto)
    else
        pycallargs(ctx, f)
    end
(b::Builtin{:call})(f, args...; kwargs...) = pycall(b.ctx, f, args...; kwargs...)

function pycallargs(ctx::Context, f)
    ans = PyNULL
    @autohdl ctx f
    ans = ctx.newhdl(ctx._c.PyObject_CallObject(ctx.cptr(f), C.PyNULL))
    @autoclosehdl ctx f
    return ans
end
function pycallargs(ctx::Context, f, args)
    ans = PyNULL
    @autohdl ctx f args
    ans = ctx.newhdl(ctx._c.PyObject_CallObject(ctx.cptr(f), ctx.cptr(args)))
    @autoclosehdl ctx f args
    return ans
end
function pycallargs(ctx::Context, f, args, kwargs)
    ans = PyNULL
    @autohdl ctx f args kwargs
    ans = ctx.newhdl(ctx._c.PyObject_Call(ctx.cptr(f), ctx.cptr(args), ctx.cptr(kwargs)))
    @autoclosehdl ctx f args kwargs
    return ans
end
(b::Builtin{:callargs})(args...) = pycallargs(b.ctx, args...)
