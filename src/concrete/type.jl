"""
    pytype(x)

The Python `type` of `x`.
"""
pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(getptr(x_))))
export pytype

"""
    pytype(name, bases, dict)

Create a new type. Equivalent to `type(name, bases, dict)` in Python.

If `bases` is not a Python object, it is converted to one using `pytuple`.

If `dict` is not a Python object, it is converted to one using `pydict`.

In order to use a Julia `Function` as an instance method, it must be wrapped into a Python
function with [`pyfunc`](@ref). Similarly, see also [`pyclassmethod`](@ref),
[`pystaticmethod`](@ref) or [`pyproperty`](@ref). In all these cases, the arguments passed
to the function always have type `Py`. See the example below.

# Example

```
Foo = pytype("Foo", (), [
    "__init__" => pyfunc(
        doc = \"\"\"
        Specify x and y to store in the Foo.

        If omitted, y defaults to None.
        \"\"\",
        function (self, x, y = nothing)
            self.x = x
            self.y = y
            return
        end,
    ),

    "__repr__" => function (self)
        return "Foo(\$(self.x), \$(self.y))"
    end |> pyfunc,

    "frompair" => pyclassmethod(
        doc = "Construct a Foo from a tuple of length two.",
        (cls, xy) -> cls(xy...),
    ),

    "hello" => pystaticmethod(
        doc = "Prints a friendly greeting.",
        (name) -> println("Hello, \$name"),
    )

    "xy" => pyproperty(
        doc = "A tuple of x and y.",
        get = (self) -> (self.x, self.y),
        set = function (self, xy)
            (x, y) = xy
            self.x = x
            self.y = y
            nothing
        end,
    ),
])
```
"""
function pytype(name, bases, dict)
    bases2 = ispy(bases) ? bases : pytuple(bases)
    dict2 = ispy(dict) ? dict : pydict(dict)
    pybuiltins.type(name, bases2, dict2)
end

pyistype(x) = pytypecheckfast(x, C.Py_TPFLAGS_TYPE_SUBCLASS)

pytypecheck(x, t) = (@autopy x t C.Py_TypeCheck(getptr(x_), getptr(t_))) == 1
pytypecheckfast(x, f) = (@autopy x C.Py_TypeCheckFast(getptr(x_), f)) == 1
