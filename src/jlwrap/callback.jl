const pywrapcallback = pynew()
const pyjlcallbacktype = pynew()

pyjlcallback_repr(self) = Py("<jl $(repr(self))>")

pyjlcallback_str(self) = Py(string(self))

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
export pyjlcallback

function pycallback(f; name=nothing, doc=nothing)
    f2 = pyjlcallback(f)
    f3 = pywrapcallback(f2)
    pydel!(f2)
    if name !== nothing
        f3.__name__ = f3.__qualname__ = name
    end
    if doc !== nothing
        f3.__doc__ = doc
    end
    return f3
end
export pycallback
