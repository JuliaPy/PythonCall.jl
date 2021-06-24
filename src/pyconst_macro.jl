"""
    @pyconst ex

Equivalent to `Py(ex)` but always returns the exact same Julia object.

That is, if `foo() = @pyconst ex` then `foo() === foo()`.

The expression `ex` is evaluated the first time the code is run.

Do not use this macro at the top level of a module. Instead, use `pynew()` and `pycopy!()`.
"""
macro pyconst(ex)
    x = pynew()
    :(ispynull($x) ? pycopy!($x, Py($(esc(ex)))) : $x)
end
export @pyconst
