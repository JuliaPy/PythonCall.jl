const pywrapcallback = pynew()
const pyjlcallbacktype = pynew()

pyjlcallback_repr(self) = Py("<jl $(repr(self))>")

pyjlcallback_str(self) = Py(sprint(print, self))

function pyjlcallback_call(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Py}, args_)
        kwargs = pyconvert(Dict{Symbol,Py}, kwargs_)
        Py(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Py}, args_)
        Py(self(args...))
    else
        Py(self())
    end
end
pyjl_handle_error_type(::typeof(pyjlcallback_call), self, exc::MethodError) = exc.f === self ? pybuiltins.TypeError : PyNULL

function init_jlwrap_callback()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class CallbackValue(ValueBase):
        __slots__ = ()
        __module__ = "juliacall"
        def __repr__(self):
            if self._jl_isnull():
                return "<jl NULL>"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlcallback_repr)))
        def __str__(self):
            if self._jl_isnull():
                return "NULL"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlcallback_str)))
        def __call__(self, *args, **kwargs):
            return self._jl_callmethod($(pyjl_methodnum(pyjlcallback_call)), args, kwargs)
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlcallbacktype, jl.CallbackValue)
    pycopy!(pywrapcallback, pybuiltins.eval("lambda f: lambda *args, **kwargs: f(*args, **kwargs)", pydict()))
end

pyjlcallback(f) = pyjl(pyjlcallbacktype, f)

"""
    pyfunc(f; name=nothing, qualname=name, doc=nothing, signature=nothing)

Wrap the callable `f` as an ordinary Python function.

The name, qualname, docstring or signature can optionally be set with `name`, `qualname`,
`doc` or `signature`.

Unlike `Py(f)` (or `pyjl(f)`), the arguments passed to `f` are always of type `Py`, i.e.
they are never converted.
"""
function pyfunc(f; name=nothing, qualname=name, doc=nothing, signature=nothing)
    f2 = ispy(f) ? f : pyjlcallback(f)
    f3 = pywrapcallback(f2)
    pydel!(f2)
    f3.__name__ = name === nothing ? "<lambda>" : name
    f3.__qualname__ = name === nothing ? "<lambda>" : qualname
    if doc !== nothing
        f3.__doc__ = doc
    end
    if signature !== nothing
        f3.__signature__ = signature
    end
    return f3
end
export pyfunc

"""
    pyclassmethod(f; ...)

Convert callable `f` to a Python class method.

If `f` is not a Python object (e.g. if `f` is a `Function`) then it is converted to one with
[`pyfunc`](@ref). In particular this means the arguments passed to `f` are always of type
`Py`. Keyword arguments are passed to `pyfunc`.
"""
pyclassmethod(f; kw...) = pybuiltins.classmethod(ispy(f) ? f : pyfunc(f; kw...))
export pyclassmethod

"""
    pystaticmethod(f; ...)

Convert callable `f` to a Python static method.

If `f` is not a Python object (e.g. if `f` is a `Function`) then it is converted to one with
[`pyfunc`](@ref). In particular this means the arguments passed to `f` are always of type
`Py`. Any keyword arguments are passed to `pyfunc`.
"""
pystaticmethod(f; kw...) = pybuiltins.staticmethod(ispy(f) ? f : pyfunc(f; kw...))
export pystaticmethod

"""
    pyproperty(; get=nothing, set=nothing, del=nothing, doc=nothing)
    pyproperty(get)

Create a Python `property` with the given getter, setter and deleter.

If `get`, `set` or `del` is not a Python object (e.g. if it is a `Function`) then it is
converted to one with [`pyfunc`](@ref). In particular this means the arguments passed to it
are always of type `Py`.
"""
pyproperty(; get=nothing, set=nothing, del=nothing, doc=nothing) =
    pybuiltins.property(
        fget = ispy(get) || get === nothing ? get : pyfunc(get),
        fset = ispy(set) || set === nothing ? set : pyfunc(set),
        fdel = ispy(del) || del === nothing ? del : pyfunc(del),
        doc = doc,
    )
pyproperty(get) = pyproperty(get=get)
export pyproperty
