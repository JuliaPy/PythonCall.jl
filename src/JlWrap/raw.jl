const pyjlrawtype = pynew()

pyjlraw_repr(self) = Py("<jl $(repr(self))>")

pyjlraw_str(self) = Py(sprint(print, self))

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))

pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

function pyjlraw_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    pyjlraw(getproperty(self, k))
end

function pyjlraw_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    v = pyconvert(Any, v_)
    setproperty!(self, k, v)
    Py(nothing)
end

pyjlraw_dir(self) = pylist(pyjl_attr_jl2py(string(k)) for k in propertynames(self, true))

function pyjlraw_call(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = pyjlraw(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = pyjlraw(self(args...))
    else
        ans = pyjlraw(self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end

pyjlraw_len(self) = Py(length(self))

function pyjlraw_getitem(self, k_::Py)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        pyjlraw(self[k...])
    else
        k = pyconvert(Any, k_)
        pyjlraw(self[k])
    end
end

function pyjlraw_setitem(self, k_::Py, v_::Py)
    v = pyconvert(Any, v_)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        self[k...] = v
    else
        k = pyconvert(Any, k_)
        self[k] = v
    end
    Py(nothing)
end

function pyjlraw_delitem(self, k_::Py)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        delete!(self, k...)
    else
        k = pyconvert(Any, k_)
        delete!(self, k)
    end
    Py(nothing)
end

pyjlraw_bool(self::Bool) = Py(self)
pyjlraw_bool(self) = (errset(pybuiltins.TypeError, "Only Julia 'Bool' can be tested for truthyness"); PyNULL)

function init_raw()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class RawValue(JlBase):
        __slots__ = ()
        def __repr__(self):
            if self._jl_isnull():
                return "<jl NULL>"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_repr)))
        def __str__(self):
            if self._jl_isnull():
                return "NULL"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_str)))
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getattr)), k)
        def __setattr__(self, k, v):
            try:
                JlBase.__setattr__(self, k, v)
            except AttributeError:
                if k.startswith("__") and k.endswith("__"):
                    raise
            else:
                return
            self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __dir__(self):
            return JlBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlraw_dir)))
        def __call__(self, *args, **kwargs):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_call)), args, kwargs)
        def __len__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_len)))
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getitem)), k)
        def __setitem__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlraw_setitem)), k, v)
        def __delitem__(self, k):
            self._jl_callmethod($(pyjl_methodnum(pyjlraw_delitem)), k)
        def __bool__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_bool)))
        def _jl_any(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjl)))
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlrawtype, jl.RawValue)
end

"""
    pyjlraw(v)

Create a Python object wrapping the Julia object `x`.

It has type `juliacall.RawValue`. This has a much more rigid "Julian" interface than `pyjl(v)`.
For example, accessing attributes or calling this object will always return a `RawValue`.
"""
pyjlraw(v) = pyjl(pyjlrawtype, v)
export pyjlraw
