const pyjlrawtype = pynew()

pyjlraw_repr(self) = Py("<jl $(repr(self))>")

pyjlraw_str(self) = Py(string(self))

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))

pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

function pyjlraw_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pyjlraw(getproperty(self, k))
end

function pyjlraw_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    v = pyconvert(Any, v_)
    setproperty!(self, k, v)
    Py(nothing)
end

pyjlraw_dir(self) = pylist(pyjl_attr_jl2py(string(k)) for k in propertynames(self, true))

function pyjlraw_call(self, args_::Py, kwargs_::Py)
    # TODO:
    # args = pyconvert(Vector{Any}, args_)
    # kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
    args = Any[]
    for a in args_
        push!(args, pyconvert(Any, a))
        pydel!(a)
    end
    kwargs = Dict{Symbol,Any}()
    for k in kwargs_
        v = kwargs_[k]
        push!(kwargs, pyconvert(Symbol, k) => pyconvert(Any, v))
        pydel!(k)
        pydel!(v)
    end
    pyjlraw(self(args...; kwargs...))
end

pyjlraw_len(self) = Py(length(self))

function pyjlraw_getitem(self, k_::Py)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
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
        delete!(self, k...)
    else
        k = pyconvert(Any, k_)
        delete!(self, k)
    end
    Py(nothing)
end

pyjlraw_bool(self::Bool) = Py(self)
pyjlraw_bool(self) = (errset(pybuiltins.TypeError, "Only Julia 'Bool' can be tested for truthyness"); pynew())

function init_jlwrap_raw()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class RawValue(ValueBase):
        __slots__ = ()
        __module__ = "juliacall"
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
                ValueBase.__setattr__(self, k, v)
            except AttributeError:
                if k.startswith("__") and k.endswith("__"):
                    raise AttributeError(k)
                else:
                    self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __dir__(self):
            return ValueBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlraw_dir)))
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
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlrawtype, jl.RawValue)
end

pyjlraw(v) = pyjl(pyjlrawtype, v)
export pyjlraw
