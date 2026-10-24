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
    unsafe_pydel(k_)
    pyjl(getproperty(self, k))
end
pyjl_handle_error_type(::typeof(pyjlany_getattr), self, exc) = pybuiltins.AttributeError
# Module attribute access is often the first call of a JuliaCall session.
precompile(pyjlany_getattr, (Module, Py))

function pyjlany_setattr(self, k_::Py, v_::Py)
    k = Symbol(pyjl_attr_py2jl(pyconvert(String, k_)))
    unsafe_pydel(k_)
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
    unsafe_pydel(args_)
    unsafe_pydel(kwargs_)
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
    unsafe_pydel(args_)
    unsafe_pydel(kwargs_)
    ans
end
pyjl_handle_error_type(::typeof(pyjlany_callback), self, exc::MethodError) =
    exc.f === self ? pybuiltins.TypeError : PyNULL

function pyjlany_getitem(self, k_::Py)
    if self isa Type
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            unsafe_pydel(k_)
            pyjl(self{k...})
        else
            k = pyconvert(Any, k_)
            pyjl(self{k})
        end
    else
        if pyistuple(k_)
            k = pyconvert(Vector{Any}, k_)
            unsafe_pydel(k_)
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
        unsafe_pydel(k_)
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
        unsafe_pydel(k_)
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
        unsafe_pydel(other_)
    else
        other = pyconvert(Any, other_)
    end
    pyjl(op.op(self, other))
end
function (op::pyjlany_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        unsafe_pydel(other_)
    else
        other = pyconvert(Any, other)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        unsafe_pydel(other2_)
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
        unsafe_pydel(other_)
    else
        other = pyconvert(Any, other_)
    end
    pyjl(op.op(other, self))
end
function (op::pyjlany_rev_op)(self, other_::Py, other2_::Py)
    if pyisjl(other_)
        other = pyjlvalue(other_)
        unsafe_pydel(other_)
    else
        other = pyconvert(Any, other)
    end
    if pyisjl(other2_)
        other2 = pyjlvalue(other2_)
        unsafe_pydel(other2_)
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
            unsafe_pydel(vo)
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
        @pyregion errset(
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
        @pyregion errset(
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
    unsafe_pydel(ndigits_)
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
        @pyregion errset(pybuiltins.StopIteration)
        PyNULL
    else
        pyjl(s[1])
    end
end

function pyjliter_next(self)
    s = iterate(self)
    if s === nothing
        @pyregion errset(pybuiltins.StopIteration)
        PyNULL
    else
        Py(s[1])
    end
end

pyjlany_hash(self) = pyint(hash(self))

const PYNUMPYDTYPE = IdDict{Type,Py}()

function pyjlany_numpy_dtype(self::Type)
    ans = get!(PYNUMPYDTYPE, self) do
        typestr, descr = pytypestrdescr(self)
        # unsupported type
        if typestr == ""
            return PyNULL
        end
        np = pyimport("numpy")
        # simple scalar type
        if pyisnull(descr)
            return np.dtype(typestr)
        end
        # We could juse use np.dtype(descr), but when there is padding, np.dtype(descr)
        # changes the names of the padding fields from "" to "f{N}". Using this other
        # dtype constructor avoids this issue and preserves the invariant:
        #   np.dtype(eltype(array)) == np.array(array).dtype
        names = []
        formats = []
        offsets = []
        for i = 1:fieldcount(self)
            nm = fieldname(self, i)
            push!(names, nm isa Integer ? "f$(nm-1)" : String(nm))
            ts, ds = pytypestrdescr(fieldtype(self, i))
            push!(formats, pyisnull(ds) ? ts : ds)
            push!(offsets, fieldoffset(self, i))
        end
        return np.dtype(
            pydict(
                names = pylist(names),
                formats = pylist(formats),
                offsets = pylist(offsets),
                itemsize = sizeof(self),
            ),
        )
    end
    if pyisnull(ans)
        @pyregion errset(pybuiltins.AttributeError, "__numpy_dtype__")
    end
    return ans
end

function init_any()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.any")
    methods = m._jl_methods
    methods["__repr__"] = pyjl_methodnum(pyjlany_repr)
    methods["__hash__"] = pyjl_methodnum(pyjlany_hash)
    methods["iterator_next"] = pyjl_methodnum(pyjliter_next)
    methods["__str__"] = pyjl_methodnum(pyjlany_str)
    methods["__getattr__"] = pyjl_methodnum(pyjlany_getattr)
    methods["__setattr__"] = pyjl_methodnum(pyjlany_setattr)
    methods["__dir__"] = pyjl_methodnum(pyjlany_dir)
    methods["__call__"] = pyjl_methodnum(pyjlany_call)
    methods["__bool__"] = pyjl_methodnum(pyjlany_bool)
    methods["__len__"] = pyjl_methodnum(pyjlany_op(length))
    methods["__getitem__"] = pyjl_methodnum(pyjlany_getitem)
    methods["__setitem__"] = pyjl_methodnum(pyjlany_setitem)
    methods["__delitem__"] = pyjl_methodnum(pyjlany_delitem)
    methods["__iter__"] = pyjl_methodnum(pyjlany_op(Iterator))
    methods["__next__"] = pyjl_methodnum(pyjlany_next)
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
    methods["__int__"] = pyjl_methodnum(pyjlany_int)
    methods["__float__"] = pyjl_methodnum(pyjlany_float)
    methods["__complex__"] = pyjl_methodnum(pyjlany_complex)
    methods["__index__"] = pyjl_methodnum(pyjlany_index)
    methods["__trunc__"] = pyjl_methodnum(pyjlany_trunc)
    methods["__floor__"] = pyjl_methodnum(pyjlany_floor)
    methods["__ceil__"] = pyjl_methodnum(pyjlany_ceil)
    methods["__round__"] = pyjl_methodnum(pyjlany_round)
    methods["__name__"] = pyjl_methodnum(pyjlany_name)
    methods["jl_display"] = pyjl_methodnum(pyjlany_display)
    methods["jl_help"] = pyjl_methodnum(pyjlany_help)
    methods["jl_eval"] = pyjl_methodnum(pyjlany_eval)
    methods["jl_to_py"] = pyjl_methodnum(Py)
    methods["jl_callback"] = pyjl_methodnum(pyjlany_callback)
    methods["_repr_mimebundle_"] = pyjl_methodnum(pyjlany_mimebundle)
    methods["__numpy_dtype__"] = pyjl_methodnum(pyjlany_numpy_dtype)

    pycopy!(pyjlanytype, jl.Jl)
    pycopy!(pyjlitertype, jl.JlIter)
end

"""
    pyjl(x)

Create a Python `juliacall.Jl` object wrapping the Julia object `x`.
"""
pyjl(v) = pyjl(pyjlanytype, v)

pyjliter(x) = pyjl(pyjlitertype, x)
