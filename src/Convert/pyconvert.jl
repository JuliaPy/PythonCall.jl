struct PyConvertRule
    id::Int
    source::String
    type::Type
    scope::Type
    func::Function
end

# Source nodes retain their first registration position, including across replacement.
mutable struct PyConvertSource
    name::String
    type::Py
    index::Int
    priority::Int
end
const PYCONVERT_RULES = PyConvertRule[]
const PYCONVERT_SOURCES = PyConvertSource[]
const PYCONVERT_PENDING = Set{String}()
const PYCONVERT_ORDER = PyConvertRule[]
const PYCONVERT_NEXT_ID = Ref(0)
const PYCONVERT_DIRTY = Ref(true)
const PYCONVERT_MODULES = Set{String}()
const PYCONVERT_PROTOCOLS = ("<arraystruct>", "<arrayinterface>", "<array>", "<buffer>")

# Normalize at registration/context construction, rather than during dispatch.
function pyconvert_source_name(name::String)
    name == "None" && return "types:NoneType"
    name in PYCONVERT_PROTOCOLS || ':' in name ? name : "builtins:" * name
end

"""
    PyConvertContext(; prefer=())

Explicit conversion context. Preferences are ordered `"module:name" => JuliaType`
pairs, resolved by attribute lookup in loaded modules. A bare name abbreviates
`"builtins:name"`, except `"None"` abbreviates `"types:NoneType"`.
Rules propagate preferences by calling `pyconvert(ctx, T, child)`
explicitly.
"""
struct PyConvertContext{P}
    prefer::P
end
PyConvertContext(; prefer=()) = PyConvertContext(map(p -> pyconvert_source_name(first(p)) => Val(last(p)), prefer))
const PYCONVERT_DEFAULT_CONTEXT = PyConvertContext()

function pyconvert_invalidate!()
    empty!.(values(PYCONVERT_RULES_CACHE))
    empty!(PYCONVERT_MODULES)
    PYCONVERT_DIRTY[] = true
end

function pyconvert_source!(name::String)
    any(s -> s.name == name, PYCONVERT_SOURCES) && return
    push!(PYCONVERT_SOURCES, PyConvertSource(name, PyNULL, length(PYCONVERT_SOURCES)+1, 0))
    name in PYCONVERT_PROTOCOLS || push!(PYCONVERT_PENDING, name)
end

"""
    pyconvert_add_rule(source, T, f; scope=T) -> id

Register a new rule. Source names use `"module:name"` (or just `"name"` for
builtins). Resolution lazily looks up attributes, including dotted paths, in
loaded modules; names need not match the type's __module__ or __qualname__.
A rule is eligible when its output intersects the requested type R and R is a
subtype of scope (or a normalized union arm is). Own the source for scope=Any,
or own the destination scope. Rules return nothing to decline, a value to succeed,
or Some(value) to explicitly succeed (required for a successful nothing/Some).
Functions may accept `(ctx, T, x)` or `(T, x)`; the former is preferred.
"""
function pyconvert_add_rule(source::String, T::Type, f::Function; scope::Type=T)
    source = pyconvert_source_name(source)
    pyconvert_source!(source)
    id = (PYCONVERT_NEXT_ID[] += 1)
    push!(PYCONVERT_RULES, PyConvertRule(id, source, T, scope, f))
    pyconvert_invalidate!()
    id
end

"""Remove a conversion rule by ID. Unknown IDs throw KeyError."""
function pyconvert_remove_rule(id::Integer)
    i = findfirst(r -> r.id == id, PYCONVERT_RULES)
    i === nothing && throw(KeyError(id))
    deleteat!(PYCONVERT_RULES, i)
    pyconvert_invalidate!()
    nothing
end

"""Replace a rule, preserving its ID and creation position."""
function pyconvert_replace_rule(id::Integer, source::String, T::Type, f::Function; scope::Type=T)
    i = findfirst(r -> r.id == id, PYCONVERT_RULES)
    i === nothing && throw(KeyError(id))
    source = pyconvert_source_name(source)
    pyconvert_source!(source)
    PYCONVERT_RULES[i] = PyConvertRule(id, source, T, scope, f)
    pyconvert_invalidate!()
    id
end

# Rule results only need a wrapper when the successful value is ambiguous.
# Public pytryconvert still wraps every success in Some.
@inline pyconvert_return(x) = x
@inline pyconvert_return(x::Union{Nothing,Some}) = Some(x)
@inline pyconvert_unconverted() = nothing
@inline pyconvert_isunconverted(x) = x === nothing
@inline pyconvert_result(::Type{T}, x::Some) where T = something(x)::T
@inline pyconvert_result(x::Some) = something(x)
@inline pyconvert_result(::Type{T}, x) where T = x::T
@inline pyconvert_result(x) = x
function _pyconvert_returntype(T::Type)
    T === Any && return Union{Nothing,Some}
    T isa Union && return Union{_pyconvert_returntype(T.a),_pyconvert_returntype(T.b)}
    isconcretetype(T) && return Union{Nothing,Some{T}}
    Union{Nothing,Some{<:T}}
end
@generated pyconvert_returntype(::Type{T}) where T = QuoteNode(_pyconvert_returntype(T))
# Avoid allocating an escaping Some for every element of an inferred container.
# The public/rule contract remains Some/nothing. Only dispatch uses this private
# sentinel, with an escape for values that happen to use our internal types.
struct PyConvertFailure end
struct PyConvertEscape{T}
    value::T
end
const PYCONVERT_FAILED = PyConvertFailure()
@inline pyconvert_pack(x) = x isa Union{PyConvertFailure,PyConvertEscape} ? PyConvertEscape(x) : x
@inline pyconvert_unpack(x) = x
@inline pyconvert_unpack(x::PyConvertEscape) = x.value
# Check before returning across the dispatch boundary, so a concrete bits value
# can use Julia's inferred return convention instead of being boxed as Any.
@inline function pyconvert_check_raw(::Type{T}, x) where T
    x === PYCONVERT_FAILED && return PYCONVERT_FAILED
    T === Any ? x : pyconvert_pack(pyconvert_unpack(x)::T)
end
@inline pyconvert_raw(::Nothing) = PYCONVERT_FAILED
@inline pyconvert_raw(x::Some) = pyconvert_pack(something(x))
@inline pyconvert_raw(x) = pyconvert_pack(x)
@inline function pyconvert_raw(::Type{T}, x) where T
    x === nothing && return PYCONVERT_FAILED
    value = x isa Some ? something(x) : x
    pyconvert_pack(value::T)
end

pyconvert_tryconvert(::Type{T}, x::T) where T = pyconvert_return(x)
function pyconvert_tryconvert(::Type{T}, x) where T
    try
        pyconvert_return(convert(T, x)::T)
    catch exc
        # Expected numeric/representation failures only. User errors propagate.
        (exc isa Union{InexactError,OverflowError} ||
         (exc isa MethodError && exc.f === convert)) || rethrow()
        nothing
    end
end

pyconvert_typename(t::Py) = "$(pygetattr(t, "__module__", "<unknown>")):$(pygetattr(t, "__qualname__", "<unknown>"))"
function pyconvert_protocol(name, t::Py)
    name == "<buffer>" && return C.PyType_CheckBuffer(t)
    name == "<array>" && return pyhasattr(t, "__array__")
    name == "<arrayinterface>" && return pyhasattr(t, "__array_interface__")
    name == "<arraystruct>" && return pyhasattr(t, "__array_struct__")
    false
end

# Never import optional modules as a side effect of conversion.
function pyconvert_resolve(name::String)
    name = pyconvert_source_name(name)
    name == "builtins:NoneType" && return pytype(pybuiltins.None)
    parts = split(name, ':'; limit=2)
    length(parts) == 2 || throw(ArgumentError("invalid Python source name: $name"))
    m = pygetitem(pyimport("sys").modules, parts[1], PyNULL)
    pyisnull(m) && return PyNULL
    for part in split(parts[2], '.')
        m = pygetattr(m, part, PyNULL)
        pyisnull(m) && return PyNULL
    end
    pyisinstance(m, pybuiltins.type) || throw(ArgumentError("$name is not a Python type"))
    m
end
function pyconvert_resolve_pending!(encountered=PyNULL)
    changed = false
    for s in PYCONVERT_SOURCES
        s.name in PYCONVERT_PENDING || continue
        t = pyconvert_resolve(s.name)
        # Types made with exec need not be exported by their named module.
        # Resolve registered real ancestors by identity, without using MRO order.
        if pyisnull(t) && !pyisnull(encountered)
            for base in encountered.__mro__
                if pyconvert_typename(base) == s.name
                    t = base
                    break
                end
            end
        end
        pyisnull(t) && continue
        s.type = t
        delete!(PYCONVERT_PENDING, s.name)
        changed = true
    end
    changed && pyconvert_invalidate!()
end

function pyconvert_subtype(a::PyConvertSource, b::PyConvertSource)
    a === b && return false
    av, bv = a.name in PYCONVERT_PROTOCOLS, b.name in PYCONVERT_PROTOCOLS
    av && return !bv && b.name == "builtins:object"
    bv && return pyconvert_protocol(b.name, a.type)
    pyissubclass(a.type, b.type) && !pyissubclass(b.type, a.type)
end

# Stable Kahn ordering. All strict subtype edges participate, not just MRO edges.
function pyconvert_toposort(edges, priorities, indices)
    remaining = collect(eachindex(priorities))
    order = Int[]
    while !isempty(remaining)
        ready = filter(i -> !any(j -> edges[j,i], remaining), remaining)
        isempty(ready) && error("cyclic pyconvert source relationships")
        i = first(sort!(ready; by=i -> (-priorities[i], indices[i])))
        push!(order, i)
        deleteat!(remaining, findfirst(==(i), remaining))
    end
    order
end
function pyconvert_order!()
    PYCONVERT_DIRTY[] || return
    sources = filter(s -> !(s.name in PYCONVERT_PENDING) &&
        (s.name in PYCONVERT_PROTOCOLS || s.name in ("builtins:object","juliacall:JlBase") ||
         any(r -> r.source == s.name, PYCONVERT_RULES)), PYCONVERT_SOURCES)
    edges = [pyconvert_subtype(a,b) for a in sources, b in sources]
    priorities = [s.name == "juliacall:JlBase" ? 30 :
                  s.name == "<buffer>" ? 10 : s.name in PYCONVERT_PROTOCOLS ? 20 : 0 for s in sources]
    # Public re-exports are the same source type, including its built-in priority.
    for i in eachindex(sources), j in eachindex(sources)
        if !pyisnull(sources[i].type) && pyis(sources[i].type,sources[j].type)
            priorities[i] = max(priorities[i],priorities[j])
        end
    end
    # Propagate maximum priority from every ancestor to every descendant.
    for _ in eachindex(sources), i in eachindex(sources), j in eachindex(sources)
        edges[i,j] && (priorities[i] = max(priorities[i], priorities[j]))
    end
    order = pyconvert_toposort(edges, priorities, [s.index for s in sources])
    empty!(PYCONVERT_ORDER)
    emitted = Set{String}()
    for i in order
        sources[i].priority = priorities[i]
        sources[i].name in emitted && continue
        aliases = [s.name for s in sources if s.name == sources[i].name ||
            (!pyisnull(s.type) && !pyisnull(sources[i].type) && pyis(s.type,sources[i].type))]
        union!(emitted,aliases)
        append!(PYCONVERT_ORDER, filter(r -> r.source in aliases, PYCONVERT_RULES))
    end
    PYCONVERT_DIRTY[] = false
end
pyconvert_scope(R, S) = R isa Union ? pyconvert_scope(R.a,S) || pyconvert_scope(R.b,S) : R <: S
pyconvert_eligible(r, R) = pyconvert_scope(R,r.scope) && typeintersect(r.type,R) !== Union{}
function pyconvert_matches(name, t)
    name in PYCONVERT_PROTOCOLS && return pyconvert_protocol(name,t)
    s = findfirst(s -> s.name == name, PYCONVERT_SOURCES)
    s === nothing && return (u = pyconvert_resolve(name); !pyisnull(u) && pyissubclass(t,u))
    u = PYCONVERT_SOURCES[s].type
    pyisnull(u) && (u = pyconvert_resolve(name))
    !pyisnull(u) && pyissubclass(t,u)
end

# Closures capture the calling convention once, for this context and target type.
function pyconvert_fix(ctx, ::Type{T}, f) where T
    if applicable(f, ctx, T, PyNULL)
        (ctx, x) -> pyconvert_raw(T,f(ctx,T,x))
    else
        (ctx, x) -> pyconvert_raw(T,f(T,x))
    end
end
struct PyConvertSelection
    type::Py # Own a reference: cached pointers can never be recycled underneath us.
    rules::Vector{Function}
    firstfunc::Any
    firsttarget::Type
end
const PYCONVERT_RULES_CACHE = Dict{Tuple{Type,Type},Dict{C.PyPtr,PyConvertSelection}}()
@generated pyconvert_rules_cache(::Type{CTX}, ::Type{T}) where {CTX,T} =
    get!(Dict{C.PyPtr,PyConvertSelection}, PYCONVERT_RULES_CACHE, (CTX,T))
function pyconvert_selection(ctx, ::Type{R}, tptr) where R
    cache = pyconvert_rules_cache(typeof(ctx), R)
    haskey(cache,tptr) && return cache[tptr]
    t = pynew(incref(tptr))
    mod = pystr(String, pygetattr(t,"__module__","<unknown>"))
    if !(mod in PYCONVERT_MODULES)
        pyconvert_resolve_pending!(t)
        push!(PYCONVERT_MODULES,mod)
    end
    pyconvert_order!()
    rules = Function[]
    firstfunc, firsttarget = nothing, Union{}
    for r in PYCONVERT_ORDER
        pyconvert_eligible(r,R) && pyconvert_matches(r.source,t) || continue
        for T in Utils.explode_union(typeintersect(r.type,R))
            T === Union{} && continue
            if isempty(rules)
                firstfunc, firsttarget = r.func, T
            end
            push!(rules,pyconvert_fix(ctx,T,r.func))
        end
    end
    cache[tptr] = PyConvertSelection(t,rules,firstfunc,firsttarget)
end

# Direct calls only when the cache proves this is the first selected rule and
# target. Registry mutations clear the proof together with the selected rules.
@inline function pyconvert_first(ctx, ::Type{T}, x, selection) where T
    f = selection.firstfunc
    if selection.firsttarget === T
        if T === Int || T === BigInt || T === Integer
            f === pyconvert_rule_int && return pyconvert_check_raw(T,pyconvert_rule_int_raw(T,x))
        elseif T === Float64
            f === pyconvert_rule_float && return pyconvert_check_raw(T,pyconvert_raw(pyconvert_rule_float(T,x)))
        elseif T === Bool
            f === pyconvert_rule_bool && return pyconvert_check_raw(T,pyconvert_raw(pyconvert_rule_bool(T,x)))
        elseif T === String
            f === pyconvert_rule_str && return pyconvert_check_raw(T,pyconvert_raw(pyconvert_rule_str(T,x)))
        elseif T === Nothing
            f === pyconvert_rule_none && return nothing
        end
    end
    pyconvert_check_raw(T,@pyregionbreak(selection.rules[1](ctx,x)))
end

function pytryconvert_plain(ctx, ::Type{T}, x::Py) where T
    selection = pyconvert_selection(ctx,T,C.Py_Type(x))
    if !isempty(selection.rules)
        ans = pyconvert_first(ctx,T,x,selection)
        ans === PYCONVERT_FAILED || return ans
        for i in 2:length(selection.rules)
            ans = @pyregionbreak(selection.rules[i](ctx,x))
            ans === PYCONVERT_FAILED || return pyconvert_check_raw(T, ans)
        end
    end
    Py <: T ? x : PYCONVERT_FAILED
end
pyconvert_preference_type(::Val{T}) where T = T
function pytryconvert_raw(ctx::PyConvertContext, ::Type{T}, x_) where T
    x = Py(x_)
    for (source, preferred) in ctx.prefer
        pyconvert_matches(source,pytype(x)) || continue
        Q = typeintersect(T,pyconvert_preference_type(preferred))
        Q === Union{} && continue
        ans = pytryconvert_plain(ctx,Q,x)
        ans === PYCONVERT_FAILED || return ans
    end
    pytryconvert_plain(ctx,T,x)
end
"""
    pytryconvert(T, x; prefer=())
    pytryconvert(ctx::PyConvertContext, T, x)

Return `Some(value)` on successful conversion and `nothing` on decline.
The successful value is a `T`; exceptions from conversion rules propagate.
"""
function pytryconvert(ctx::PyConvertContext, ::Type{T}, x) where T
    @pyregion begin
        ans = pytryconvert_raw(ctx,T,x)
        ans === PYCONVERT_FAILED && return nothing
        Some(pyconvert_unpack(ans)::T)::pyconvert_returntype(T)
    end
end
pytryconvert(::Type{T}, x; prefer=()) where T = pytryconvert(PyConvertContext(;prefer),T,x)

"""Inspect source-matching rules in global order, including destination eligibility."""
function pyconvert_rules(R::Type, x)
    t = pytype(Py(x))
    pyconvert_resolve_pending!()
    pyconvert_order!()
    [(; r.id, r.source, output=r.type, r.scope,
        priority=only(s.priority for s in PYCONVERT_SOURCES if s.name==r.source),
        order=i, eligible=pyconvert_eligible(r,R))
     for (i,r) in enumerate(PYCONVERT_ORDER) if pyconvert_matches(r.source,t)]
end
function pyconvert_preferred_type(t::Py)
    pyconvert_resolve_pending!()
    pyconvert_order!()
    for r in PYCONVERT_ORDER
        pyconvert_eligible(r,Any) && pyconvert_matches(r.source,t) && return r.type
    end
    Py
end

"""
    @pyconvert(T, x, [onfail])
    @pyconvert(ctx, T, x, onfail)

Convert a child in a rule, returning failure from the rule unless onfail is given.
"""
macro pyconvert(args...)
    if length(args) == 4
        ctx,T,x,onfail = args
    else
        T,x = args[1:2]
        ctx = :($PYCONVERT_DEFAULT_CONTEXT)
        onfail = length(args)==3 ? args[3] : :(return nothing)
    end
    quote
        @pyregion begin
            target = $(esc(T))
            ans = pytryconvert($(esc(ctx)),target,$(esc(x)))
            ans === nothing ? $(esc(onfail)) : pyconvert_result(target,ans)
        end
    end
end

"""
    pyconvert(T, x, [default]; prefer=())
    pyconvert(ctx::PyConvertContext, T, x, [default])

Convert to T, preferring native containers. Source-specific preferences run in
order before ordinary conversion. Failure throws, or returns default if supplied.
"""
function pyconvert(ctx::PyConvertContext, ::Type{T}, x) where T
    @pyregion begin
        ans = pytryconvert_raw(ctx,T,x)
        ans === PYCONVERT_FAILED && error("cannot convert Python object to Julia '$T'")
        pyconvert_unpack(ans)::T
    end
end
function pyconvert(ctx::PyConvertContext, ::Type{T}, x, default) where T
    @pyregion begin
        ans = pytryconvert_raw(ctx,T,x)
        ans === PYCONVERT_FAILED ? default : pyconvert_unpack(ans)::T
    end
end
pyconvert(::Type{T}, x; prefer=()) where T = pyconvert(PyConvertContext(;prefer),T,x)
pyconvert(::Type{T}, x, default; prefer=()) where T = pyconvert(PyConvertContext(;prefer),T,x,default)
function pyconvertarg(::Type{T}, x, name) where T
    @autopy x @pyconvert T x_ begin
        errset(pybuiltins.TypeError,"Cannot convert argument '$name' to a Julia '$T'")
        pythrow()
    end
end

function init_pyconvert()
    pyimport("types")
    pyimport("numbers")
    pyimport("collections.abc")
    pyimport("io")
    pyconvert_source!("builtins:object")
    pyconvert_add_rule("types:NoneType", Nothing, pyconvert_rule_none; scope=Any)
    pyconvert_add_rule("builtins:bool", Bool, pyconvert_rule_bool; scope=Any)
    pyconvert_add_rule("builtins:float", Float64, pyconvert_rule_float; scope=Any)
    pyconvert_add_rule(
        "builtins:complex",
        Complex{Float64},
        pyconvert_rule_complex;
        scope = Any,
    )
    pyconvert_add_rule("numbers:Integral", Integer, pyconvert_rule_int; scope=Any)
    pyconvert_add_rule("builtins:str", String, pyconvert_rule_str; scope=Any)
    pyconvert_add_rule(
        "builtins:bytes",
        Base.CodeUnits{UInt8,String},
        pyconvert_rule_bytes;
        scope = Any,
    )
    pyconvert_add_rule(
        "builtins:range",
        StepRange{<:Integer,<:Integer},
        pyconvert_rule_range;
        scope = Any,
    )
    pyconvert_add_rule(
        "numbers:Rational",
        Rational{<:Integer},
        pyconvert_rule_fraction;
        scope = Any,
    )
    pyconvert_add_rule("builtins:tuple", NamedTuple, pyconvert_rule_iterable; scope=Any)
    pyconvert_add_rule("builtins:tuple", Tuple, pyconvert_rule_iterable; scope=Any)
    pyconvert_add_rule("datetime:datetime", DateTime, pyconvert_rule_datetime; scope=Any)
    pyconvert_add_rule("datetime:date", Date, pyconvert_rule_date; scope=Any)
    pyconvert_add_rule("datetime:time", Time, pyconvert_rule_time; scope=Any)
    pyconvert_add_rule(
        "datetime:timedelta",
        Microsecond,
        pyconvert_rule_timedelta;
        scope = Any,
    )
    pyconvert_add_rule(
        "builtins:BaseException",
        PyException,
        pyconvert_rule_exception;
        scope = Any,
    )

    pyconvert_add_rule("types:NoneType", Missing, pyconvert_rule_none)
    pyconvert_add_rule("builtins:bool", Number, pyconvert_rule_bool)
    # Real/Complex ABC membership does not promise lossless __float__/__complex__.
    # Integral and Rational sources have their own exact representation rules.
    pyconvert_add_rule("builtins:float", Number, pyconvert_rule_float)
    pyconvert_add_rule("builtins:float", Nothing, pyconvert_rule_float)
    pyconvert_add_rule("builtins:float", Missing, pyconvert_rule_float)
    pyconvert_add_rule("builtins:complex", Number, pyconvert_rule_complex)
    pyconvert_add_rule("numbers:Integral", Number, pyconvert_rule_int)
    pyconvert_add_rule("builtins:str", Symbol, pyconvert_rule_str)
    pyconvert_add_rule("builtins:str", Char, pyconvert_rule_str)
    pyconvert_add_rule("builtins:bytes", Vector{UInt8}, pyconvert_rule_bytes)
    pyconvert_add_rule(
        "builtins:range",
        UnitRange{<:Integer},
        pyconvert_rule_range;
        scope = AbstractRange,
    )
    pyconvert_add_rule("numbers:Rational", Number, pyconvert_rule_fraction)
    pyconvert_add_rule(
        "collections.abc:Iterable",
        Vector,
        pyconvert_rule_iterable,
    )
    pyconvert_add_rule("collections.abc:Iterable", Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Iterable", Pair, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Iterable", Set, pyconvert_rule_iterable)
    pyconvert_add_rule(
        "collections.abc:Sequence",
        Vector,
        pyconvert_rule_iterable,
    )
    pyconvert_add_rule("collections.abc:Sequence", Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Set", Set, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Mapping", Dict, pyconvert_rule_mapping)
    pyconvert_add_rule(
        "datetime:timedelta",
        Millisecond,
        pyconvert_rule_timedelta,
    )
    pyconvert_add_rule("datetime:timedelta", Second, pyconvert_rule_timedelta)
    pyconvert_add_rule("datetime:timedelta", Nanosecond, pyconvert_rule_timedelta)

end
