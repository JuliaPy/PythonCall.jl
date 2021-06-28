const pyjlanytype = pynew()

pyjlany_repr(self) = Py("<jl $(repr(self))>")

pyjlany_str(self) = Py(string(self))

function pyjlany_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    Py(getproperty(self, k))
end
pyjl_handle_error_type(::typeof(pyjlany_getattr), self, exc) = pybuiltins.AttributeError

function pyjlany_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    v = pyconvert(Any, v_)
    setproperty!(self, k, v)
    Py(nothing)
end
pyjl_handle_error_type(::typeof(pyjlany_setattr), self, exc) = pybuiltins.AttributeError

pyjlany_dir(self) = pylist(pyjl_attr_jl2py(string(k)) for k in propertynames(self, true))

function pyjlany_call(self, args_::Py, kwargs_::Py)
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
    Py(self(args...; kwargs...))
end
pyjl_handle_error_type(::typeof(pyjlany_call), self, exc) = exc isa MethodError && exc.f === self ? pybuiltins.TypeError : PyNULL

pyjlany_len(self) = Py(length(self))
pyjl_handle_error_type(::typeof(pyjlany_len), self, exc) = exc isa MethodError && exc.f === length ? pybuiltins.TypeError : PyNULL

pyjlany_getitem(self, k_::Py) =
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        Py(self[k...])
    else
        k = pyconvert(Any, k_)
        Py(self[k])
    end
pyjl_handle_error_type(::typeof(pyjlany_getitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

function pyjlany_setitem(self, k_::Py, v_::Py)
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
pyjl_handle_error_type(::typeof(pyjlany_setitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

function pyjlany_delitem(self, k_::Py)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        delete!(self, k...)
    else
        k = pyconvert(Any, k_)
        delete!(self, k)
    end
    Py(nothing)
end
pyjl_handle_error_type(::typeof(pyjlany_delitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

pyjlany_iter(self) = pyjl(Iterator(self))

function init_jlwrap_any()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class AnyValue(ValueBase):
        __slots__ = ()
        __module__ = "juliacall"
        def __repr__(self):
            if self._jl_isnull():
                return "<jl NULL>"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_repr)))
        def __str__(self):
            if self._jl_isnull():
                return "NULL"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_str)))
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_getattr)), k)
        def __setattr__(self, k, v):
            try:
                ValueBase.__setattr__(self, k, v)
            except AttributeError:
                if k.startswith("__") and k.endswith("__"):
                    raise AttributeError(k)
                else:
                    self._jl_callmethod($(pyjl_methodnum(pyjlany_setattr)), k, v)
        def __dir__(self):
            return ValueBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlany_dir)))
        def __call__(self, *args, **kwargs):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_call)), args, kwargs)
        def __len__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_len)))
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_getitem)), k)
        def __setitem__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlany_setitem)), k, v)
        def __delitem__(self, k):
            self._jl_callmethod($(pyjl_methodnum(pyjlany_delitem)), k)
        def __iter__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_iter)))
        def _jl_raw(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw)))
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlanytype, jl.AnyValue)
end

pyjl(v) = pyjl(pyjlanytype, v)
