const pywrapcallback = pynew()

pyjlcallback(f) = pyjlany(f).jl_callback

"""
    pyfunc(f; [name], [qualname], [doc], [signature])

Wrap the callable `f` as an ordinary Python function.

The name, qualname, docstring or signature can optionally be set with `name`, `qualname`,
`doc` or `signature`. If either of `name` or `qualname` are given, the other is inferred.

Unlike `Py(f)` (or `pyjl(f)`), the arguments passed to `f` are always of type `Py`, i.e.
they are never converted.
"""
function pyfunc(f; name=nothing, qualname=nothing, doc=nothing, signature=nothing, wrap=pywrapcallback)
    f2 = ispy(f) ? f : pyjlcallback(f)
    if wrap isa Pair
        wrapargs, wrapfunc = wrap
    else
        wrapargs, wrapfunc = (), wrap
    end
    if wrapfunc === pywrapcallback && pyisnull(pywrapcallback)
        pycopy!(pywrapcallback, pybuiltins.eval("lambda f: lambda *args, **kwargs: f(*args, **kwargs)", pydict()))
    end
    if wrapfunc isa AbstractString
        f3 = pybuiltins.eval(wrapfunc, pydict())(f2, wrapargs...)
    else
        f3 = wrapfunc(f2, wrapargs...)
    end
    if name === nothing && qualname !== nothing
        name = split(qualname, '.')[end]
    elseif name !== nothing && qualname === nothing
        qualname = name
    end
    f3.__name__ = name === nothing ? "<lambda>" : name
    f3.__qualname__ = qualname === nothing ? "<lambda>" : qualname
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
[`pyfunc`](@ref PythonCall.pyfunc). In particular this means the arguments passed to `f` are always of type
`Py`. Keyword arguments are passed to `pyfunc`.
"""
pyclassmethod(f; kw...) = pybuiltins.classmethod(ispy(f) ? f : pyfunc(f; kw...))
export pyclassmethod

"""
    pystaticmethod(f; ...)

Convert callable `f` to a Python static method.

If `f` is not a Python object (e.g. if `f` is a `Function`) then it is converted to one with
[`pyfunc`](@ref PythonCall.pyfunc). In particular this means the arguments passed to `f` are always of type
`Py`. Any keyword arguments are passed to `pyfunc`.
"""
pystaticmethod(f; kw...) = pybuiltins.staticmethod(ispy(f) ? f : pyfunc(f; kw...))
export pystaticmethod

"""
    pyproperty(; get=nothing, set=nothing, del=nothing, doc=nothing, ...)
    pyproperty(get, set=nothing, del=nothing; doc=nothing, ...)

Create a Python `property` with the given getter, setter and deleter.

If `get`, `set` or `del` is not a Python object (e.g. if it is a `Function`) then it is
converted to one with [`pyfunc`](@ref PythonCall.pyfunc). In particular this means the arguments passed to it
are always of type `Py`.
"""
pyproperty(; get=nothing, set=nothing, del=nothing, doc=nothing, kw...) =
    pybuiltins.property(
        fget = ispy(get) || get === nothing ? get : pyfunc(get; kw...),
        fset = ispy(set) || set === nothing ? set : pyfunc(set; kw...),
        fdel = ispy(del) || del === nothing ? del : pyfunc(del; kw...),
        doc = doc,
    )
pyproperty(get, set=nothing, del=nothing; doc=nothing, kw...) = pyproperty(; get=get, set=set, del=del, doc=doc, kw...)
export pyproperty
