"""
    pytype(x)

The Python `type` of `x`.
"""
pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(getptr(x_))))
export pytype

"""
    pytype(name, bases, dict)

Create a new type. Equivalent to `type(name, bases, dict)` in Python.

See [`pyclass`](@ref) for a more convenient syntax.
"""
pytype(name, bases, dict) = pybuiltins.type(name, ispy(bases) ? bases : pytuple(bases), ispy(dict) ? dict : pydict(dict))

"""
    pyclass(name, bases=(); members...)

Construct a new Python type with the given name, bases and members.

The `bases` may be a Python type or a tuple of Python types.

Any `members` which are Julia functions are interpreted as instance methods (equivalent to
wrapping the function in [`pyfunc`](@ref)). To create class methods, static methods or
properties, wrap the function in [`pyclassmethod`](@ref), [`pystaticmethod`](@ref) or
[`pyproperty`](@ref).

Note that the arguments to any method or property are passed as `Py`, i.e. they are not
converted first.

# Example

```
Foo = pyclass("Foo",
    __init__ = function (self, x, y = nothing)
        self.x = x
        self.y = y
        nothing
    end,

    __repr__ = function (self)
        "Foo(\$(self.x), \$(self.y))"
    end,

    frompair = function (cls, xy)
        cls(xy...)
    end
    |> pyclassmethod,

    hello = function (name)
        println("Hello, \$name")
    end
    |> pystaticmethod,

    xy = pyproperty(
        get = function (self)
            (self.x, self.y)
        end,
        set = function (self, xy)
            (x, y) = xy
            self.x = x
            self.y = y
            nothing
        end,
    ),
)
```
"""
function pyclass(name, bases=(); members...)
    bases2 = ispy(bases) && pyistype(bases) ? pytuple((bases,)) : pytuple(bases)
    members2 = pydict(pystr(k) => v isa Function ? pyfunc(v) : Py(v) for (k, v) in members)
    pytype(name, bases2, members2)
end
export pyclass

pyistype(x) = pytypecheckfast(x, C.Py_TPFLAGS_TYPE_SUBCLASS)

pytypecheck(x, t) = (@autopy x t C.Py_TypeCheck(getptr(x_), getptr(t_))) == 1
pytypecheckfast(x, f) = (@autopy x C.Py_TypeCheckFast(getptr(x_), f)) == 1
