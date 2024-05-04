const pyjlanytype = pynew()

pyjlany(x) = pyjl(pyjlanytype, x)

# pyjlany_repr(self) = Py("<jl $(repr(self))>")
function pyjlany_repr(self)
    str = repr(MIME("text/plain"), self; context=IOContext(devnull, :limit=>true, :displaysize=>(23,80)))
    # type = self isa Function ? "Function" : self isa Type ? "Type" : nameof(typeof(self))
    sep = '\n' in str ? '\n' : ' '
    Py("Julia:$sep$str"::String)
end

# Note: string(self) doesn't always return a String
pyjlany_str(self) = Py(sprint(print, self)::String)

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))

pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

function pyjlany_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    pyjlany(getproperty(self, k))
end
pyjl_handle_error_type(::typeof(pyjlany_getattr), self, exc) = pybuiltins.AttributeError

function pyjlany_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    v = pyconvert(Any, v_)
    setproperty!(self, k, v)
    Py(nothing)
end
pyjl_handle_error_type(::typeof(pyjlany_setattr), self, exc) = pybuiltins.AttributeError

function pyjlany_dir(self)
    ks = Symbol[]
    if self isa Module
        append!(ks, names(self, all = true, imported = true))
        for m in ccall(:jl_module_usings, Any, (Any,), self)::Vector
            append!(ks, names(m))
        end
    else
        append!(propertynames(self, true))
    end
    pylist(pyjl_attr_jl2py(string(k)) for k in ks)
end

function pyjlany_call(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = pyjlany(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = pyjlany(self(args...))
    else
        ans = pyjlany(self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_call), self, exc) = exc isa MethodError && exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_getitem(self, k_::Py)
    if self isa Type
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            pydel!(k_)
            pyjlany(self{k...})
        else
            k = pyconvert(Any, k_)
            pyjlany(self{k})
        end    
    else
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            pydel!(k_)
            pyjlany(self[k...])
        else
            k = pyconvert(Any, k_)
            pyjlany(self[k])
        end
    end
end
pyjl_handle_error_type(::typeof(pyjlany_getitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

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
pyjl_handle_error_type(::typeof(pyjlany_setitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

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
pyjl_handle_error_type(::typeof(pyjlany_delitem), self, exc) = exc isa BoundsError ? pybuiltins.IndexError : exc isa KeyError ? pybuiltins.KeyError : PyNULL

pyjlany_contains(self, v::Py) = Py((@pyconvert(eltype(self), v, return Py(false)) in self)::Bool)
pyjl_handle_error_type(::typeof(pyjlany_contains), self, exc) = exc isa MethodError && exc.f === in ? pybuiltins.TypeError : PyNULL

struct pyjlany_op{OP}
    op :: OP
end
(op::pyjlany_op)(self) = pyjlany(op.op(self))
function (op::pyjlany_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = pyconvert(Any, other_)
    end
    pyjlany(op.op(self, other))
end
function (op::pyjlany_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = pyconvert(Any, other)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        pydel!(other2_)
    end
    pyjlany(op.op(self, other, other2))
end
pyjl_handle_error_type(op::pyjlany_op, self, exc) = exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

struct pyjlany_rev_op{OP}
    op :: OP
end
function (op::pyjlany_rev_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = pyconvert(Any, other_)
    end
    pyjlany(op.op(other, self))
end
function (op::pyjlany_rev_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = pyconvert(Any, other)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        pydel!(other2_)
    end
    pyjlany(op.op(other, self, other2))
end
pyjl_handle_error_type(op::pyjlany_rev_op, self, exc) = exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

pyjlany_name(self) = Py(string(nameof(self))::String)
pyjl_handle_error_type(::typeof(pyjlany_name), self, exc) = exc isa MethodError && exc.f === nameof ? pybuiltins.AttributeError : PyNULL

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
            show(IOContext(io, :limit=>true), MIME(m), self)
            v = take!(io)
            ans[m] = vo = istextmime(m) ? pystr(String(v)) : pybytes(v)
            pydel!(vo)
        catch err
            # silently skip anything that didn't work
        end
    end
    return ans
end

pyjlany_eval(self::Module, expr::Py) = Py(Base.eval(self, Meta.parseall(strip(pyconvert(String, expr)))))

pyjl_handle_error_type(::typeof(pyjlany_eval), self, exc) = exc isa MethodError ? pybuiltins.TypeError : PyNULL

pyjlany_int(self) = pyint(convert(Integer, self))

pyjl_handle_error_type(::typeof(pyjlany_int), self, exc) = exc isa MethodError ? pybuiltins.TypeError : PyNULL

pyjlany_float(self) = pyfloat(convert(AbstractFloat, self))

pyjl_handle_error_type(::typeof(pyjlany_float), self, exc) = exc isa MethodError ? pybuiltins.TypeError : PyNULL

pyjlany_complex(self) = pycomplex(convert(Complex, self))

pyjl_handle_error_type(::typeof(pyjlany_complex), self, exc) = exc isa MethodError ? pybuiltins.TypeError : PyNULL

function pyjlany_index(self)
    if self isa Integer
        pyint(self)
    else
        errset(pybuiltins.TypeError, "Only Julia 'Integer' values can be used as Python indices, not '$(typeof(self))'")
    end
end

function pyjlany_bool(self)
    if self isa Bool
        pybool(self)
    else
        errset(pybuiltins.TypeError, "Only Julia 'Bool' values can be tested for truthyness, not '$(typeof(self))'")
        PyNULL
    end
end

function init_any()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class Jl(JlBase):
        __slots__ = ()
        def __repr__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_repr)))
        def __str__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_str)))
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_getattr)), k)
        def __setattr__(self, k, v):
            try:
                JlBase.__setattr__(self, k, v)
            except AttributeError:
                if k.startswith("__") and k.endswith("__"):
                    raise
            else:
                return
            self._jl_callmethod($(pyjl_methodnum(pyjlany_setattr)), k, v)
        def __dir__(self):
            return JlBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlany_dir)))
        def __call__(self, *args, **kwargs):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_call)), args, kwargs)
        def __bool__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_bool)))
        def __len__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(length))))
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_getitem)), k)
        def __setitem__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlany_setitem)), k, v)
        def __delitem__(self, k):
            self._jl_callmethod($(pyjl_methodnum(pyjlany_delitem)), k)
        def __iter__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(Iterator))))
        def __reversed__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(reverse))))
        def __contains__(self, v):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_contains)), v)
        def __pos__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(+))))
        def __neg__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(-))))
        def __abs__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(abs))))
        def abs(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(abs))))
        def __invert__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(~))))
        def __add__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(+))), other)
        def __sub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(-))), other)
        def __mul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(*))), other)
        def __truediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(/))), other)
        def __floordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(÷))), other)
        def __mod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(%))), other)
        def __pow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(powermod))), other, modulo)
        def __lshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(<<))), other)
        def __rshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(>>))), other)
        def __and__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(&))), other)
        def __xor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(⊻))), other)
        def __or__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(|))), other)
        def __radd__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(+))), other)
        def __rsub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(-))), other)
        def __rmul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(*))), other)
        def __rtruediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(/))), other)
        def __rfloordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(÷))), other)
        def __rmod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(%))), other)
        def __rpow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(powermod))), other, modulo)
        def __rlshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(<<))), other)
        def __rrshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(>>))), other)
        def __rand__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(&))), other)
        def __rxor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(⊻))), other)
        def __ror__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_rev_op(|))), other)
        def __eq__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(==))), other)
        def __ne__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(!=))), other)
        def __le__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(≤))), other)
        def __lt__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(<))), other)
        def __ge__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(≥))), other)
        def __gt__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(>))), other)
        def __hash__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_op(hash))))
        def __int__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_int)))
        def __float__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_float)))
        def __complex__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_complex)))
        def __index__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_index)))
        @property
        def __name__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_name)))
        def jl_display(self, mime=None):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_display)), mime)
        def jl_help(self, mime=None):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_help)), mime)
        def jl_eval(self, expr):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_eval)), expr)
        def _repr_mimebundle_(self, include=None, exclude=None):
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_mimebundle)), include, exclude)
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlanytype, jl.Jl)
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
export pyjl

"""
    pyjltype(x)

The subtype of `juliacall.Jl` which the Julia object `x` is wrapped as by `pyjl(x)`.

Overload `pyjltype(::T)` to define a custom conversion for your type `T`.
"""
pyjltype(::Any) = pyjlanytype
export pyjltype
