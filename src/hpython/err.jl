iserr(py::Context) = py._c.PyErr_Occurred() != C.PyNULL

errclear(py::Context) = py._c.PyErr_Clear()
(b::Builtin{:errclear})() = errclear(b.ctx)

function errset(py::Context, t)
    @autohdl py t
    py._c.PyErr_SetNone(py.cptr(t))
    @autoclosehdl py t
end
function errset(py::Context, t, v)
    @autohdl py t v
    py._c.PyErr_SetObject(py.cptr(t), py.cptr(v))
    @autoclosehdl py t v
end
function errset(py::Context, t, v::String)
    @autohdl py t
    py._c.PyErr_SetString(py.cptr(t), v)
    @autoclosehdl py t
end
(b::Builtin{:errset})(args...) = errset(b.ctx, args...)

function errget(py::Context, normalize::Bool=false)
    t = Ref(C.PyNULL)
    v = Ref(C.PyNULL)
    b = Ref(C.PyNULL)
    py._c.PyErr_Fetch(t, v, b)
    if normalize
        py._c.PyErr_NormalizeException(t, v, b)
    end
    (py.newhdl(t[]), py.newhdl(v[]), py.newhdl(b[]))
end
(b::Builtin{:errget})(args...) = errget(b.ctx, args...)
