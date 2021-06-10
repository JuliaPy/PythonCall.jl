pyobject(py::Context, x::PyAnyHdl) = py.duphdl(x)
pyobject(py::Context, x::Builtin) = py.duphdl(py.autohdl(x))
pyobject(py::Context, x::Nothing) = pyobject(py, py.None)
pyobject(py::Context, x::Bool) = x ? pyobject(py, py.True) : pyobject(py, py.False)
pyobject(py::Context, x::Union{String,SubString{String}}) = py.str(x)
pyobject(py::Context, x::Tuple) = py.tuple(x)
function pyobject(py::Context, x)
    py.errset(py.TypeError, "cannot convert this Julia '$(typeof(x))' to a Python object")
    PyNULL
end
(py::Context)(x) = pyobject(py, x)

function pyhasattr(py::Context, x, k)
    ans = BoolOrErr()
    @autohdl py x k
    ans = BoolOrErr(py._c.PyObject_HasAttr(py.cptr(x), py.cptr(k)))
    @autoclosehdl py x k
    return ans
end
function pyhasattr(py::Context, x::PyHdl, k::PyHdl)
    # this special case can return Bool instead of BoolOrErr
    py._c.PyObject_HasAttr(py.cptr(x), py.cptr(k)) != 0
end
(f::Builtin{:hasattr})(x, k) = pyhasattr(f.ctx, x, k)

function pygetattr(py::Context, x, k)
    ans = PyNULL
    @autohdl py x k
    ans = py.newhdl(py._c.PyObject_GetAttr(py.cptr(x), py.cptr(k)))
    @autoclosehdl py x k
    return ans
end
(f::Builtin{:getattr})(x, k) = pygetattr(f.ctx, x, k)

function pysetattr(py::Context, x, k, v)
    ans = VoidOrErr()
    @autohdl py x k v
    ans = VoidOrErr(py._c.PyObject_SetAttr(py.cptr(x), py.cptr(k), py.cptr(v)))
    @autoclosehdl py x k v
    return ans
end
(f::Builtin{:setattr})(x, k, v) = pysetattr(f.ctx, x, k, v)

function pyascii(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyObject_ASCII(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
pyascii(::Type{T}, py::Context, x) where {T} = pystr_convert(T, py, pyascii(py, x).auto)
(f::Builtin{:ascii})(x) = pyascii(f.ctx, x)
(f::Builtin{:ascii})(::Type{T}, x) where {T} = pyascii(T, f.ctx, x)

function pyrepr(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyObject_Repr(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
pyrepr(::Type{T}, py::Context, x) where {T} = pystr_convert(T, py, pyrepr(py, x).auto)
(f::Builtin{:repr})(x) = pyrepr(f.ctx, x)
(f::Builtin{:repr})(::Type{T}, x) where {T} = pyrepr(T, f.ctx, x)
