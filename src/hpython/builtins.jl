function pyimport(py::Context, m)
    ans = PyNULL
    @autohdl py m
    ans = py.newhdl(py._c.PyImport_Import(py.cptr(m)))
    @autoclosehdl py m
    return ans
end
(f::Builtin{:import})(m) = pyimport(f.ctx, m)

pycall(py::Context, f, args...; kwargs...) =
    if !isempty(kwargs)
        error("not implemented: keyword arguments")
    elseif !isempty(args)
        pycallargs(py, f, py.tuple(args).auto)
    else
        pycallargs(py, f)
    end
(b::Builtin{:call})(f, args...; kwargs...) = pycall(b.ctx, f, args...; kwargs...)

function pycallargs(py::Context, f)
    ans = PyNULL
    @autohdl py f
    ans = py.newhdl(py._c.PyObject_CallObject(py.cptr(f), C.PyNULL))
    @autoclosehdl py f
    return ans
end
function pycallargs(py::Context, f, args)
    ans = PyNULL
    @autohdl py f args
    ans = py.newhdl(py._c.PyObject_CallObject(py.cptr(f), py.cptr(args)))
    @autoclosehdl py f args
    return ans
end
function pycallargs(py::Context, f, args, kwargs)
    ans = PyNULL
    @autohdl py f args kwargs
    ans = py.newhdl(py._c.PyObject_Call(py.cptr(f), py.cptr(args), py.cptr(kwargs)))
    @autoclosehdl py f args kwargs
    return ans
end
(b::Builtin{:callargs})(args...) = pycallargs(b.ctx, args...)
