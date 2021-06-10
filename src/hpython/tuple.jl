pytuple(py::Context) = py.newhdl(py._c.PyTuple_New(0))
pytuple(py::Context, xs::PyAnyHdl) = py.call(py.tuple, xs)
function pytuple(py::Context, xs)
    ans = py.newhdl(py._c.PyTuple_New(length(xs)))
    py.iserr(ans) && return PyNULL
    for (i,x) in enumerate(xs)
        x isa PyAutoHdl && error("cannot convert tuples containing PyAutoHdl to Python tuples")
        xh = py(x)
        if py.iserr(xh)
            py.closehdl(ans)
            return PyNULL
        end
        err = py._c.PyTuple_SetItem(py.cptr(ans), i-1, py.stealcptr(xh))
        if err == -1
            py.closehdl(ans)
            return PyNULL
        end
    end
    return ans
end
(b::Builtin{:tuple})(args...) = pytuple(b.ctx, args...)
