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

function pyjlraw_call_nogil(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = pyjlraw(GIL.@unlock self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = pyjlraw(GIL.@unlock self(args...))
    else
        ans = pyjlraw(GIL.@unlock self())
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
pyjlraw_bool(self) =
    (errset(pybuiltins.TypeError, "Only Julia 'Bool' can be tested for truthyness"); PyNULL)

function init_raw()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.raw")
    methods = m._jl_methods
    methods["__repr__"] = pyjl_methodnum(pyjlraw_repr)
    methods["__str__"] = pyjl_methodnum(pyjlraw_str)
    methods["__getattr__"] = pyjl_methodnum(pyjlraw_getattr)
    methods["__setattr__"] = pyjl_methodnum(pyjlraw_setattr)
    methods["__dir__"] = pyjl_methodnum(pyjlraw_dir)
    methods["__call__"] = pyjl_methodnum(pyjlraw_call)
    methods["__len__"] = pyjl_methodnum(pyjlraw_len)
    methods["__getitem__"] = pyjl_methodnum(pyjlraw_getitem)
    methods["__setitem__"] = pyjl_methodnum(pyjlraw_setitem)
    methods["__delitem__"] = pyjl_methodnum(pyjlraw_delitem)
    methods["__bool__"] = pyjl_methodnum(pyjlraw_bool)
    methods["_jl_any"] = pyjl_methodnum(pyjl)
    methods["_jl_call_nogil"] = pyjl_methodnum(pyjlraw_call_nogil)
    pycopy!(pyjlrawtype, jl.RawValue)
end

"""
    pyjlraw(v)

Create a Python object wrapping the Julia object `x`.

It has type `juliacall.RawValue`. This has a much more rigid "Julian" interface than `pyjl(v)`.
For example, accessing attributes or calling this object will always return a `RawValue`.
"""
pyjlraw(v) = pyjl(pyjlrawtype, v)
