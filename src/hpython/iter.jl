"""
    pyiter(py, x)

Produce a Python iterator from `x`.

Equivalent to `iter(x)` in Python.
"""
function pyiter(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyObject_GetIter(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
(b::Builtin{:iter})(x) = pyiter(b.ctx, x)

"""
    pynext(py, x)

Get the next value from Python iterator `x`.

Equivalent to `next(x)` in Python but with less error checking.
"""
function pynext(py::Context, x)
    ans = PyNULL
    @autohdl py x
    ans = py.newhdl(py._c.PyIter_Next(py.cptr(x)))
    @autoclosehdl py x
    return ans
end
(b::Builtin{:next})(x) = pynext(b.ctx, x)

"""
    pyfor(f, py, x)

Call `f(v)` for each `v` in Python object `x`.

`v` is automatically closed.

`f(v)` must be a `Nothing`, `Bool` or `BoolOrErr` or `VoidOrErr`: `nothing` or `false` means continue, `true` means break.

`f` must not throw. It may signal a Python error by returning `BoolOrErr()` or `VoidOrErr()`.

Returns `BoolOrErr`, which is true if the loop was broken, or false if iteration reached the end.
"""
function pyfor(f, py::Context, x)
    ans = BoolOrErr()
    i = py.iter(x)
    py.iserr(i) && return PyNULL
    while true
        v = py.next(i)
        if !py.iserr(v)
            # non-NULL value
            res = f(v)::Union{Nothing,Bool,BoolOrErr,VoidOrErr}
            py.closehdl(v)
            if py.iserr(res)
                # error
                ans = BoolOrErr()
                break
            elseif value(py, res) === true
                # true means break
                ans = BoolOrErr(true)
                break
            else
                # nothing or false means continue
                continue
            end
        elseif !py.iserr()
            # NULL but no error means we reached the end of the iterator
            ans = BoolOrErr(false)
            break
        else
            # error
            ans = BoolOrErr()
            break
        end
    end
    py.closehdl(i)
    return ans
end
(b::Builtin{:for})(f, x) = pyfor(f, b.ctx, x)
