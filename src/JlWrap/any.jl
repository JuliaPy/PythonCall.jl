const pyjlanytype = pynew()
const pyjlitertype = pynew()

# pyjlany_repr(self) = Py("<jl $(repr(self))>")
function pyjlany_repr(self)
    str = repr(
        MIME("text/plain"),
        self;
        context = IOContext(devnull, :limit => true, :displaysize => (23, 80)),
    )
    # type = self isa Function ? "Function" : self isa Type ? "Type" : nameof(typeof(self))
    sep = '\n' in str ? '\n' : ' '
    Py("$(sep)$(str)"::String)
end

# Note: string(self) doesn't always return a String
pyjlany_str(self) = Py(sprint(print, self)::String)

pyjl_attr_py2jl(k::String) = replace(k, r"_[b]+$" => (x -> "!"^(length(x) - 1)))

pyjl_attr_jl2py(k::String) = replace(k, r"!+$" => (x -> "_" * "b"^length(x)))

function pyjlany_getattr(self, k_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    pyjl(getproperty(self, k))
end
pyjl_handle_error_type(::typeof(pyjlany_getattr), self, exc) = pybuiltins.AttributeError

function pyjlany_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    pydel!(k_)
    v = pyconvert(Any, v_)
    if self isa Module && !isdefined(self, k)
        # Fix for https://github.com/JuliaLang/julia/pull/54678
        Base.Core.eval(self, Expr(:global, k))
    end
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
        append!(ks, propertynames(self, true))
    end
    pylist(pyjl_attr_jl2py(string(k)) for k in ks)
end

function pyjlany_call(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = pyjl(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = pyjl(self(args...))
    else
        ans = pyjl(self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_call), self, exc) =
    exc isa MethodError && exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_callback(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Py}, args_)
        kwargs = pyconvert(Dict{Symbol,Py}, kwargs_)
        ans = Py(self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Py}, args_)
        ans = Py(self(args...))
    else
        ans = Py(self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_callback), self, exc::MethodError) =
    exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_call_nogil(self, args_::Py, kwargs_::Py)
    if pylen(kwargs_) > 0
        args = pyconvert(Vector{Any}, args_)
        kwargs = pyconvert(Dict{Symbol,Any}, kwargs_)
        ans = pyjl(GIL.@unlock self(args...; kwargs...))
    elseif pylen(args_) > 0
        args = pyconvert(Vector{Any}, args_)
        ans = pyjl(GIL.@unlock self(args...))
    else
        ans = pyjl(GIL.@unlock self())
    end
    pydel!(args_)
    pydel!(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_call_nogil), self, exc::MethodError) =
    exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_getitem(self, k_::Py)
    if self isa Type
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            pydel!(k_)
            pyjl(self{k...})
        else
            k = pyconvert(Any, k_)
            pyjl(self{k})
        end
    else
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            pydel!(k_)
            pyjl(self[k...])
        else
            k = pyconvert(Any, k_)
            pyjl(self[k])
        end
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

pyjlany_contains(self, v::Py) =
    Py((@pyconvert(eltype(self), v, return Py(false)) in self)::Bool)
pyjl_handle_error_type(::typeof(pyjlany_contains), self, exc::MethodError) =
    exc.f === in ? pybuiltins.TypeError : PyNULL

struct pyjlany_op{OP}
    op::OP
end
(op::pyjlany_op)(self) = pyjl(op.op(self))
function (op::pyjlany_op)(self, other_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        pydel!(other_)
    else
        other = pyconvert(Any, other_)
    end
    pyjl(op.op(self, other))
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
    pyjl(op.op(self, other, other2))
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
    else
        other = pyconvert(Any, other_)
    end
    pyjl(op.op(other, self))
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
    pyjl(op.op(other, self, other2))
end
pyjl_handle_error_type(op::pyjlany_rev_op, self, exc) =
    exc isa MethodError && exc.f === op.op ? pybuiltins.TypeError : PyNULL

pyjlany_name(self) = Py(string(nameof(self))::String)
pyjl_handle_error_type(::typeof(pyjlany_name), self, exc::MethodError) =
    exc.f === nameof ? pybuiltins.AttributeError : PyNULL

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
        # hack: the relevant methods of Docs.doc are actually
        # in REPL, so we load it dynamically if needed
        @eval Main using REPL
        doc = invokelatest(Docs.doc, self)
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

pyjlany_eval(self::Module, expr::Py) =
    pyjl(Base.eval(self, Meta.parseall(strip(pyconvert(String, expr)))))
pyjl_handle_error_type(::typeof(pyjlany_eval), self, exc::MethodError) =
    pybuiltins.TypeError

pyjlany_int(self) = pyint(convert(Integer, self))
pyjl_handle_error_type(::typeof(pyjlany_int), self, exc::MethodError) = pybuiltins.TypeError

pyjlany_float(self) = pyfloat(convert(AbstractFloat, self))
pyjl_handle_error_type(::typeof(pyjlany_float), self, exc::MethodError) =
    pybuiltins.TypeError

pyjlany_complex(self) = pycomplex(convert(Complex, self))
pyjl_handle_error_type(::typeof(pyjlany_complex), self, exc::MethodError) =
    pybuiltins.TypeError

function pyjlany_index(self)
    if self isa Integer
        pyint(self)
    else
        errset(
            pybuiltins.TypeError,
            "Only Julia 'Integer' values can be used as Python indices, not '$(typeof(self))'",
        )
        PyNULL
    end
end

function pyjlany_bool(self)
    if self isa Bool
        pybool(self)
    else
        errset(
            pybuiltins.TypeError,
            "Only Julia 'Bool' values can be tested for truthyness, not '$(typeof(self))'",
        )
        PyNULL
    end
end

pyjlany_trunc(self) = pyint(trunc(Integer, self))
pyjl_handle_error_type(::typeof(pyjlany_trunc), self, exc::MethodError) =
    pybuiltins.TypeError

pyjlany_floor(self) = pyint(floor(Integer, self))
pyjl_handle_error_type(::typeof(pyjlany_floor), self, exc::MethodError) =
    pybuiltins.TypeError

pyjlany_ceil(self) = pyint(ceil(Integer, self))
pyjl_handle_error_type(::typeof(pyjlany_ceil), self, exc::MethodError) =
    pybuiltins.TypeError

pyjlany_round(self) = pyint(round(Integer, self))
function pyjlany_round(self, ndigits_::Py)
    ndigits = pyconvertarg(Int, ndigits_, "ndigits")
    pydel!(ndigits_)
    pyjl(round(self; digits = ndigits))
end
pyjl_handle_error_type(::typeof(pyjlany_round), self, exc::MethodError) =
    pybuiltins.TypeError

mutable struct Iterator
    value::Any
    state::Any
    started::Bool
    finished::Bool
end

Iterator(x) = Iterator(x, nothing, false, false)
Iterator(x::Iterator) = x

function Base.iterate(x::Iterator, ::Nothing = nothing)
    if x.finished
        s = nothing
    elseif x.started
        s = iterate(x.value, x.state)
    else
        s = iterate(x.value)
    end
    if s === nothing
        x.finished = true
        nothing
    else
        x.started = true
        x.state = s[2]
        (s[1], nothing)
    end
end

function pyjlany_next(self)
    s = iterate(self)
    if s === nothing
        errset(pybuiltins.StopIteration)
        PyNULL
    else
        pyjl(s[1])
    end
end

function pyjliter_next(self)
    s = iterate(self)
    if s === nothing
        errset(pybuiltins.StopIteration)
        PyNULL
    else
        Py(s[1])
    end
end

pyjlany_hash(self) = pyint(hash(self))

function init_any()
    jl = pyjuliacallmodule
    pybuiltins.exec(
        pybuiltins.compile(
            """
$("\n"^(@__LINE__()-1))
class JlBase2(JlBase):
    __slots__ = ()
    def __repr__(self):
        t = type(self)
        if t is Jl:
            name = "Julia"
        else:
            name = t.__name__
        return name + ":" + self._jl_callmethod($(pyjl_methodnum(pyjlany_repr)))
class JlIter(JlBase2):
    __slots__ = ()
    def __iter__(self):
        return self
    def __hash__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_hash)))
    def __next__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjliter_next)))
class Jl(JlBase2):
    __slots__ = ()
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
    def __next__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_next)))
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
    def __hash__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_hash)))
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
    def __int__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_int)))
    def __float__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_float)))
    def __complex__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_complex)))
    def __index__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_index)))
    def __trunc__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_trunc)))
    def __floor__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_floor)))
    def __ceil__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_ceil)))
    def __round__(self, ndigits=None):
        if ndigits is None:
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_round)))
        else:
            return self._jl_callmethod($(pyjl_methodnum(pyjlany_round)), ndigits)
    @property
    def __name__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_name)))
    def jl_display(self, mime=None):
        '''Display this, optionally specifying the MIME type.'''
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_display)), mime)
    def jl_help(self, mime=None):
        '''Show help for this Julia object.'''
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_help)), mime)
    def jl_eval(self, expr):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_eval)), expr)
    def jl_to_py(self):
        return self._jl_callmethod($(pyjl_methodnum(Py)))
    def jl_callback(self, *args, **kwargs):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_callback)), args, kwargs)
    def jl_call_nogil(self, *args, **kwargs):
        '''Call this with the given arguments but with the GIL disabled.
        
        WARNING: This function must not interact with Python at all without re-acquiring
        the GIL.
        '''
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_call_nogil)), args, kwargs)
    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_mimebundle)), include, exclude)
""",
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
    pycopy!(pyjlanytype, jl.Jl)
    pycopy!(pyjlitertype, jl.JlIter)
end

"""
    pyjl(x)

Create a Python `juliacall.Jl` object wrapping the Julia object `x`.
"""
pyjl(v) = pyjl(pyjlanytype, v)

pyjliter(x) = pyjl(pyjlitertype, x)
