const pyjlanytype = pynew()

# pyjlany_repr(self) = Py("<jl $(repr(self))>")
function pyjlany_repr(self)
    str = repr(
        MIME("text/plain"),
        self;
        context = IOContext(devnull, :limit => true, :displaysize => (23, 80)),
    )
    # type = self isa Function ? "Function" : self isa Type ? "Type" : nameof(typeof(self))
    sep = '\n' in str ? '\n' : ' '
    Py("Julia:$sep$str")
end

# Note: string(self) doesn't always return a String
pyjlany_str(self) = Py(sprint(print, self))

function pyjlany_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    Py(getproperty(self, k))
end
pyjl_handle_error_type(::typeof(pyjlany_getattr), self, exc) = pybuiltins.AttributeError
# `jl.<name>` on `juliacall.Main` is often the first call of a JuliaCall session.
precompile(pyjlany_getattr, (Module, Py))

function pyjlany_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    v = pyconvert(Any, v_)
    if self isa Module && !isdefined(self, k)
        # Fix for https://github.com/JuliaLang/julia/pull/54678
        @eval self (global $k = $v)
    else
        setproperty!(self, k, v)
    end
    Py(nothing)
end
pyjl_handle_error_type(::typeof(pyjlany_setattr), self, exc) = pybuiltins.AttributeError

pyjlany_dir(self) = pylist(pyjl_attr_jl2py(string(k)) for k in propertynames(self, true))

function pyjlany_call(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = Py(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = Py(self(args...))
    else
        ans = Py(self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_call), self, exc) =
    exc isa MethodError && exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_call_nogil(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = Py(GIL.@unlock self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = Py(GIL.@unlock self(args...))
    else
        ans = Py(GIL.@unlock self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_call_nogil), self, exc) =
    exc isa MethodError && exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_getitem(self, k_::Py)
    if pyistuple(k_)
        k = pyconvert(Vector{Any}, k_)
        pydel!(k_)
        Py(self[k...])
    else
        k = pyconvert(Any, k_)
        Py(self[k])
    end
end
pyjl_handle_error_type(::typeof(pyjlany_getitem), self, exc) =
    exc isa BoundsError ? pybuiltins.IndexError :
    exc isa KeyError ? pybuiltins.KeyError : PyNULL

function pyjlany_setitem(self, k_::Py, v_::Py)
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
pyjl_handle_error_type(::typeof(pyjlany_setitem), self, exc) =
    exc isa BoundsError ? pybuiltins.IndexError :
    exc isa KeyError ? pybuiltins.KeyError : PyNULL

function pyjlany_delitem(self, k_::Py)
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
pyjl_handle_error_type(::typeof(pyjlany_delitem), self, exc) =
    exc isa BoundsError ? pybuiltins.IndexError :
    exc isa KeyError ? pybuiltins.KeyError : PyNULL

pyjlany_contains(self, v::Py) = Py(@pyconvert(eltype(self), v, return Py(false)) in self)
pyjl_handle_error_type(::typeof(pyjlany_contains), self, exc) =
    exc isa MethodError && exc.f === in ? pybuiltins.TypeError : PyNULL

struct pyjlany_op{OP}
    op::OP
end
(op::pyjlany_op)(self) = Py(op.op(self))
function (op::pyjlany_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
        Py(op.op(self, other))
    else
        pybuiltins.NotImplemented
    end
end
function (op::pyjlany_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_) && pyisjl(other2_)
        other = pyjlvalue(other_)
        other2 = pyjlvalue(other2_)
        pydel!(other_)
        pydel!(other2_)
        Py(op.op(self, other, other2))
    else
        pybuiltins.NotImplemented
    end
end
pyjl_handle_error_type(op::pyjlany_op, self, exc) =
    exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

struct pyjlany_rev_op{OP}
    op::OP
end
function (op::pyjlany_rev_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
        Py(op.op(other, self))
    else
        pybuiltins.NotImplemented
    end
end
function (op::pyjlany_rev_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_) && pyisjl(other2_)
        other = pyjlvalue(other_)
        other2 = pyjlvalue(other2_)
        pydel!(other_)
        pydel!(other2_)
        Py(op.op(other, self, other2))
    else
        pybuiltins.NotImplemented
    end
end
pyjl_handle_error_type(op::pyjlany_rev_op, self, exc) =
    exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

pyjlany_name(self) = Py(string(nameof(self)))
pyjl_handle_error_type(::typeof(pyjlany_name), self, exc) =
    exc isa MethodError && exc.f === nameof ? pybuiltins.AttributeError : PyNULL

function pyjlany_display(self, mime_::Py)
    mime = pyconvertarg(Union{Nothing,String}, mime_, "mime")
    x = Utils.ExtraNewline(self)
    if mime === nothing
        display(x)
    else
        display(mime, x)
    end
    Py(nothing)
end

function pyjlany_help(self, mime_::Py)
    mime = pyconvertarg(Union{Nothing,String}, mime_, "mime")
    doc = Docs.getdoc(self)
    if doc === nothing
        doc = Docs.doc(self)
    end
    x = Utils.ExtraNewline(doc)
    if mime === nothing
        display(x)
    else
        display(mime, x)
    end
    Py(nothing)
end

function pyjlany_mimebundle(self, include::Py, exclude::Py)
    # TODO: use include/exclude
    mimes = Utils.mimes_for(self)
    # make the bundle
    ans = pydict()
    for m in mimes
        try
            io = IOBuffer()
            show(IOContext(io, :limit => true), MIME(m), self)
            v = take!(io)
            ans[m] = vo = istextmime(m) ? pystr(String(v)) : pybytes(v)
            pydel!(vo)
        catch err
            # silently skip anything that didn't work
        end
    end
    return ans
end

function init_any()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.any")
    methods = m._jl_methods
    methods["__repr__"] = pyjl_methodnum(pyjlany_repr)
    methods["__str__"] = pyjl_methodnum(pyjlany_str)
    methods["__getattr__"] = pyjl_methodnum(pyjlany_getattr)
    methods["__setattr__"] = pyjl_methodnum(pyjlany_setattr)
    methods["__dir__"] = pyjl_methodnum(pyjlany_dir)
    methods["__call__"] = pyjl_methodnum(pyjlany_call)
    methods["__len__"] = pyjl_methodnum(pyjlany_op(length))
    methods["__getitem__"] = pyjl_methodnum(pyjlany_getitem)
    methods["__setitem__"] = pyjl_methodnum(pyjlany_setitem)
    methods["__delitem__"] = pyjl_methodnum(pyjlany_delitem)
    methods["__iter__"] = pyjl_methodnum(pyjlany_op(Iterator))
    methods["__reversed__"] = pyjl_methodnum(pyjlany_op(reverse))
    methods["__contains__"] = pyjl_methodnum(pyjlany_contains)
    methods["__pos__"] = pyjl_methodnum(pyjlany_op(+))
    methods["__neg__"] = pyjl_methodnum(pyjlany_op(-))
    methods["__abs__"] = pyjl_methodnum(pyjlany_op(abs))
    methods["__invert__"] = pyjl_methodnum(pyjlany_op(~))
    methods["__add__"] = pyjl_methodnum(pyjlany_op(+))
    methods["__sub__"] = pyjl_methodnum(pyjlany_op(-))
    methods["__mul__"] = pyjl_methodnum(pyjlany_op(*))
    methods["__truediv__"] = pyjl_methodnum(pyjlany_op(/))
    methods["__floordiv__"] = pyjl_methodnum(pyjlany_op(÷))
    methods["__mod__"] = pyjl_methodnum(pyjlany_op(%))
    methods["__pow__"] = pyjl_methodnum(pyjlany_op(^))
    methods["__pow_modulo__"] = pyjl_methodnum(pyjlany_op(powermod))
    methods["__lshift__"] = pyjl_methodnum(pyjlany_op(<<))
    methods["__rshift__"] = pyjl_methodnum(pyjlany_op(>>))
    methods["__and__"] = pyjl_methodnum(pyjlany_op(&))
    methods["__xor__"] = pyjl_methodnum(pyjlany_op(⊻))
    methods["__or__"] = pyjl_methodnum(pyjlany_op(|))
    methods["__radd__"] = pyjl_methodnum(pyjlany_rev_op(+))
    methods["__rsub__"] = pyjl_methodnum(pyjlany_rev_op(-))
    methods["__rmul__"] = pyjl_methodnum(pyjlany_rev_op(*))
    methods["__rtruediv__"] = pyjl_methodnum(pyjlany_rev_op(/))
    methods["__rfloordiv__"] = pyjl_methodnum(pyjlany_rev_op(÷))
    methods["__rmod__"] = pyjl_methodnum(pyjlany_rev_op(%))
    methods["__rpow__"] = pyjl_methodnum(pyjlany_rev_op(^))
    methods["__rpow_modulo__"] = pyjl_methodnum(pyjlany_rev_op(powermod))
    methods["__rlshift__"] = pyjl_methodnum(pyjlany_rev_op(<<))
    methods["__rrshift__"] = pyjl_methodnum(pyjlany_rev_op(>>))
    methods["__rand__"] = pyjl_methodnum(pyjlany_rev_op(&))
    methods["__rxor__"] = pyjl_methodnum(pyjlany_rev_op(⊻))
    methods["__ror__"] = pyjl_methodnum(pyjlany_rev_op(|))
    methods["__eq__"] = pyjl_methodnum(pyjlany_op(==))
    methods["__ne__"] = pyjl_methodnum(pyjlany_op(!=))
    methods["__le__"] = pyjl_methodnum(pyjlany_op(≤))
    methods["__lt__"] = pyjl_methodnum(pyjlany_op(<))
    methods["__ge__"] = pyjl_methodnum(pyjlany_op(≥))
    methods["__gt__"] = pyjl_methodnum(pyjlany_op(>))
    methods["__hash__"] = pyjl_methodnum(pyjlany_op(hash))
    methods["__name__"] = pyjl_methodnum(pyjlany_name)
    methods["_jl_raw"] = pyjl_methodnum(pyjlraw)
    methods["_jl_display"] = pyjl_methodnum(pyjlany_display)
    methods["_jl_help"] = pyjl_methodnum(pyjlany_help)
    methods["_jl_call_nogil"] = pyjl_methodnum(pyjlany_call_nogil)
    methods["_repr_mimebundle_"] = pyjl_methodnum(pyjlany_mimebundle)
    pycopy!(pyjlanytype, jl.AnyValue)
end

"""
    pyjl([t=pyjltype(x)], x)

Create a Python object wrapping the Julia object `x`.

If `x` is mutable, then mutating the returned object also mutates `x`, and vice versa.

Its Python type is normally inferred from the type of `x`, but can be specified with `t`.

For example if `x` is an `AbstractVector` then the object will have type `juliacall.VectorValue`.
This object will satisfy the Python sequence interface, so for example uses 0-up indexing.

To define a custom conversion for your type `T`, overload `pyjltype(::T)`.
"""
pyjl(v) = pyjl(pyjltype(v), v)

"""
    pyjltype(x)

The subtype of `juliacall.AnyValue` which the Julia object `x` is wrapped as by `pyjl(x)`.

Overload `pyjltype(::T)` to define a custom conversion for your type `T`.
"""
pyjltype(::Any) = pyjlanytype
