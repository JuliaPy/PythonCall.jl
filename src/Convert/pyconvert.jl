struct PyConvertRule
    tname::String
    type::Type
    scope::Type
    func::Function
    order::Int
end

const PYCONVERT_RULES = Dict{String,Vector{PyConvertRule}}()
const PYCONVERT_RULE_ORDER = Ref{Int}(0)
const PYCONVERT_EXTRATYPES = Py[]

"""
    pyconvert_add_rule(func::Function, tname::String, ::Type{T}, ::Type{S}=T) where {T,S}

Add a new conversion rule for `pyconvert`.

### Arguments

- `tname` is a string of the form `"__module__:__qualname__"` identifying a Python type `t`,
  such as `"builtins:dict"` or `"sympy.core.symbol:Symbol"`. This rule only applies to
  Python objects of this type.
- `T` is a Julia type, such that this rule only applies when the target type intersects
  with `T`.
- `S` is a Julia type, such that this rule only applies when the target type is a subtype
  of `S` (or a union whose components include a subtype of `S`).
- `func` is the function implementing the rule.

When `pyconvert(R, x)` is called, all rules such that `typeintersect(T, R) != Union{}`
and `pyisinstance(x, t)` are considered. It also requires that `R <: S`, or if `R` is a
union then at least one component satisfies this property. These rules are sorted first
by the specificity of `t` (strict subclassing only) then by the order they were added.
The rules are tried in turn until one succeeds.

### Implementing `func`

`func` is called as `func(S, x::Py)` for some `S <: T`.

It must return one of:
- `pyconvert_return(ans)` where `ans` is the result of the conversion (and must be an `S`).
- `pyconvert_unconverted()` if the conversion was not possible (e.g. converting a `list` to
  `Vector{Int}` might fail if some of the list items are not integers).

The target type `S` is never a union or the empty type, i.e. it is always a data type or
union-all.

"""
function pyconvert_add_rule(func::Function, pytypename::String, type::Type, scope::Type = type)
    @nospecialize type func scope
    type <: scope || error("pyconvert rule must satisfy T <: S")
    order = (PYCONVERT_RULE_ORDER[] += 1)
    push!(
        get!(Vector{PyConvertRule}, PYCONVERT_RULES, pytypename),
        PyConvertRule(pytypename, type, scope, func, order),
    )
    empty!.(values(PYCONVERT_RULES_CACHE))
    empty!(PYCONVERT_PREFERRED_TYPE)
    return
end

# Alternative ways to represent the result of conversion.
if true
    # Returns either the result or Unconverted().
    struct Unconverted end
    @inline pyconvert_return(x) = x
    @inline pyconvert_unconverted() = Unconverted()
    @inline pyconvert_returntype(::Type{T}) where {T} = Union{T,Unconverted}
    @inline pyconvert_isunconverted(r) = r === Unconverted()
    @inline pyconvert_result(::Type{T}, r) where {T} = r::T
elseif false
    # Stores the result in PYCONVERT_RESULT.
    # This is global state, probably best avoided.
    const PYCONVERT_RESULT = Ref{Any}(nothing)
    @inline pyconvert_return(x) = (PYCONVERT_RESULT[] = x; true)
    @inline pyconvert_unconverted() = false
    @inline pyconvert_returntype(::Type{T}) where {T} = Bool
    @inline pyconvert_isunconverted(r::Bool) = !r
    @inline pyconvert_result(::Type{T}, r::Bool) where {T} =
        (ans = PYCONVERT_RESULT[]::T; PYCONVERT_RESULT[] = nothing; ans)
else
    # Same as the previous scheme, but with special handling for bits types.
    # This is global state, probably best avoided.
    const PYCONVERT_RESULT = Ref{Any}(nothing)
    const PYCONVERT_RESULT_ISBITS = Ref{Bool}(false)
    const PYCONVERT_RESULT_TYPE = Ref{Type}(Union{})
    const PYCONVERT_RESULT_BITSLEN = 1024
    const PYCONVERT_RESULT_BITS = fill(0x00, PYCONVERT_RESULT_BITSLEN)
    function pyconvert_return(x::T) where {T}
        if isbitstype(T) && sizeof(T) ≤ PYCONVERT_RESULT_BITSLEN
            unsafe_store!(Ptr{T}(pointer(PYCONVERT_RESULT_BITS)), x)
            PYCONVERT_RESULT_ISBITS[] = true
            PYCONVERT_RESULT_TYPE[] = T
        else
            PYCONVERT_RESULT[] = x
            PYCONVERT_RESULT_ISBITS[] = false
        end
        return true
    end
    @inline pyconvert_unconverted() = false
    @inline pyconvert_returntype(::Type{T}) where {T} = Bool
    @inline pyconvert_isunconverted(r::Bool) = !r
    function pyconvert_result(::Type{T}, r::Bool) where {T}
        if isbitstype(T)
            if sizeof(T) ≤ PYCONVERT_RESULT_BITSLEN
                @assert PYCONVERT_RESULT_ISBITS[]
                @assert PYCONVERT_RESULT_TYPE[] == T
                return unsafe_load(Ptr{T}(pointer(PYCONVERT_RESULT_BITS)))::T
            end
        elseif PYCONVERT_RESULT_ISBITS[]
            t = PYCONVERT_RESULT_TYPE[]
            @assert isbitstype(t)
            @assert sizeof(t) ≤ PYCONVERT_RESULT_BITSLEN
            @assert t <: T
            @assert isconcretetype(t)
            return unsafe_load(Ptr{t}(pointer(PYCONVERT_RESULT_BITS)))::T
        end
        # general case
        ans = PYCONVERT_RESULT[]::T
        PYCONVERT_RESULT[] = nothing
        return ans::T
    end
end

pyconvert_result(r) = pyconvert_result(Any, r)

pyconvert_tryconvert(::Type{T}, x::T) where {T} = pyconvert_return(x)
pyconvert_tryconvert(::Type{T}, x) where {T} =
    try
        pyconvert_return(convert(T, x)::T)
    catch
        pyconvert_unconverted()
    end

function pyconvert_typename(t::Py)
    m = pygetattr(t, "__module__", "<unknown>")
    n = pygetattr(t, "__name__", "<name>")
    return "$m:$n"
end

pyconvert_is_special_tname(tname::String) =
    tname in ("<arraystruct>", "<arrayinterface>", "<array>", "<buffer>")

function _pyconvert_collect_supertypes(pytype::Py)
    seen = Set{C.PyPtr}()
    queue = Py[pytype]
    types = Py[]
    while !isempty(queue)
        t = pop!(queue)
        ptr = getptr(t)
        ptr ∈ seen && continue
        push!(seen, ptr)
        push!(types, t)
        if pyhasattr(t, "__bases__")
            append!(queue, (Py(b) for b in t.__bases__))
        end
    end
    return types
end

function _pyconvert_get_rules(pytype::Py)
    typemap = Dict{String,Py}()
    tnames = Set{String}()

    function add_type!(t::Py)
        tname = pyconvert_typename(t)
        haskey(typemap, tname) || (typemap[tname] = t)
        push!(tnames, tname)
        return nothing
    end

    for t in _pyconvert_collect_supertypes(pytype)
        add_type!(t)
        pyhasattr(t, "__array_struct__") && push!(tnames, "<arraystruct>")
        pyhasattr(t, "__array_interface__") && push!(tnames, "<arrayinterface>")
        pyhasattr(t, "__array__") && push!(tnames, "<array>")
        (C.PyType_CheckBuffer(t) != 0) && push!(tnames, "<buffer>")
    end

    for xtype in PYCONVERT_EXTRATYPES
        pyissubclass(pytype, xtype) || continue
        for t in _pyconvert_collect_supertypes(xtype)
            add_type!(t)
        end
    end

    rules = PyConvertRuleInfo[
        PyConvertRuleInfo(
            tname,
            get(typemap, tname, PyNULL),
            rule.type,
            rule.scope,
            rule.func,
            rule.order,
        ) for tname in tnames for rule in get!(Vector{PyConvertRule}, PYCONVERT_RULES, tname)
    ]

    return rules, typemap
end

struct PyConvertRuleInfo
    tname::String
    pytype::Py
    type::Type
    scope::Type
    func::Function
    order::Int
end

const PYCONVERT_PREFERRED_TYPE = Dict{Py,Type}()

pyconvert_preferred_type(pytype::Py) =
    get!(PYCONVERT_PREFERRED_TYPE, pytype) do
        if pyissubclass(pytype, pybuiltins.int)
            Union{Int,BigInt}
        else
            pyconvert_get_rules_info(Any, pytype)[1].type
        end
    end

function pyconvert_get_rules(type::Type, pytype::Py)
    @nospecialize type

    rules = pyconvert_get_rules_info(type, pytype)

    @debug "pyconvert" type rules
    return Function[pyconvert_fix(rule.type, rule.func) for rule in rules]
end

function pyconvert_get_rules_info(type::Type, pytype::Py)
    @nospecialize type
    rules, typemap = _pyconvert_get_rules(pytype)
    rules = _pyconvert_filter_rules(type, rules)
    return _pyconvert_order_rules(rules, typemap)
end

pyconvert_fix(::Type{T}, func) where {T} = x -> func(T, x)

const PYCONVERT_RULES_CACHE = Dict{Type,Dict{C.PyPtr,Vector{Function}}}()

@generated pyconvert_rules_cache(::Type{T}) where {T} =
    get!(Dict{C.PyPtr,Vector{Function}}, PYCONVERT_RULES_CACHE, T)

function _pyconvert_type_in_scope(::Type{R}, ::Type{S}) where {R,S}
    if R isa Union
        any(_pyconvert_type_in_scope(T, S) for T in Utils.explode_union(R))
    else
        R <: S
    end
end

function _pyconvert_filter_rules(type::Type, rules::Vector{PyConvertRuleInfo})
    @nospecialize type
    filtered = PyConvertRuleInfo[]
    for rule in rules
        T = typeintersect(rule.type, type)
        T == Union{} && continue
        _pyconvert_type_in_scope(type, rule.scope) || continue
        for U in Utils.explode_union(T)
            U == Union{} && continue
            push!(filtered, PyConvertRuleInfo(rule.tname, rule.pytype, U, rule.scope, rule.func, rule.order))
        end
    end

    filtered = [
        rule for (i, rule) in enumerate(filtered) if !any(
            (rule.func === filtered[j].func) && (rule.type <: filtered[j].type) for j = 1:(i - 1)
        )
    ]
    return filtered
end

function _pyconvert_tname_subclass(t1::String, t2::String, typemap::Dict{String,Py})
    t1 == t2 && return false
    py1 = get(typemap, t1, PyNULL)
    py2 = get(typemap, t2, PyNULL)

    if t2 == "<buffer>"
        return (t1 != "<buffer>") && !pyisnull(py1) && (C.PyType_CheckBuffer(py1) != 0)
    elseif t2 == "<array>"
        return (t1 != "<array>") && !pyisnull(py1) && pyhasattr(py1, "__array__")
    elseif t2 == "<arrayinterface>"
        return (t1 != "<arrayinterface>") && !pyisnull(py1) && pyhasattr(py1, "__array_interface__")
    elseif t2 == "<arraystruct>"
        return (t1 != "<arraystruct>") && !pyisnull(py1) && pyhasattr(py1, "__array_struct__")
    elseif pyisnull(py1) || pyisnull(py2)
        return false
    else
        return pyissubclass(py1, py2)
    end
end

function _pyconvert_order_rules(rules::Vector{PyConvertRuleInfo}, typemap::Dict{String,Py})
    incoming = Dict{Int,Vector{Int}}()
    for i in eachindex(rules)
        incoming[i] = Int[]
    end
    for (i, ri) in pairs(rules), (j, rj) in pairs(rules)
        (i == j) && continue
        if _pyconvert_tname_subclass(ri.tname, rj.tname, typemap)
            push!(incoming[j], i)
        end
    end

    ordered = PyConvertRuleInfo[]
    remaining = collect(keys(incoming))
    while !isempty(remaining)
        available = [i for i in remaining if isempty(incoming[i])]
        isempty(available) && error("pyconvert rule ordering cycle detected")
        sort!(available, by = i -> rules[i].order)
        append!(ordered, rules[available])
        for i in available
            delete!(incoming, i)
        end
        filter!(i -> haskey(incoming, i), remaining)
        for deps in values(incoming)
            filter!(i -> i ∉ available, deps)
        end
    end
    return ordered
end

function pyconvert_rule_fast(::Type{T}, x::Py) where {T}
    if T isa Union
        a = pyconvert_rule_fast(T.a, x)::pyconvert_returntype(T.a)
        pyconvert_isunconverted(a) || return a
        b = pyconvert_rule_fast(T.b, x)::pyconvert_returntype(T.b)
        pyconvert_isunconverted(b) || return b
    elseif (T == Nothing) | (T == Missing)
        pyisnone(x) && return pyconvert_return(T())
    elseif (T == Bool)
        pyisFalse(x) && return pyconvert_return(false)
        pyisTrue(x) && return pyconvert_return(true)
    elseif (T == Int) | (T == BigInt)
        pyisint(x) && return pyconvert_rule_int(T, x)
    elseif (T == Float64)
        pyisfloat(x) && return pyconvert_return(T(pyfloat_asdouble(x)))
    elseif (T == ComplexF64)
        pyiscomplex(x) && return pyconvert_return(T(pycomplex_ascomplex(x)))
    elseif (T == String) | (T == Char) | (T == Symbol)
        pyisstr(x) && return pyconvert_rule_str(T, x)
    elseif (T == Vector{UInt8}) | (T == Base.CodeUnits{UInt8,String})
        pyisbytes(x) && return pyconvert_rule_bytes(T, x)
    elseif (T <: StepRange) | (T <: UnitRange)
        pyisrange(x) && return pyconvert_rule_range(T, x)
    end
    pyconvert_unconverted()
end

function pytryconvert(::Type{T}, x_) where {T}
    # Convert the input to a Py
    x = Py(x_)

    # We can optimize the conversion for some types by overloading pytryconvert_fast.
    # It MUST give the same results as via the slower route using rules.
    ans1 = pyconvert_rule_fast(T, x)::pyconvert_returntype(T)
    pyconvert_isunconverted(ans1) || return ans1

    # get rules from the cache
    # TODO: we should hold weak references and clear the cache if types get deleted
    tptr = C.Py_Type(x)
    trules = pyconvert_rules_cache(T)
    rules = get!(trules, tptr) do
        t = pynew(incref(tptr))
        ans = pyconvert_get_rules(T, t)::Vector{Function}
        pydel!(t)
        ans
    end

    # apply the rules
    for rule in rules
        ans2 = rule(x)::pyconvert_returntype(T)
        pyconvert_isunconverted(ans2) || return ans2
    end

    return pyconvert_unconverted()
end

"""
    @pyconvert(T, x, [onfail])

Convert the Python object `x` to a `T`.

On failure, evaluates to `onfail`, which defaults to `return pyconvert_unconverted()` (mainly useful for writing conversion rules).
"""
macro pyconvert(T, x, onfail = :(return $pyconvert_unconverted()))
    quote
        T = $(esc(T))
        x = $(esc(x))
        ans = pytryconvert(T, x)
        if pyconvert_isunconverted(ans)
            $(esc(onfail))
        else
            pyconvert_result(T, ans)
        end
    end
end

"""
    pyconvert(T, x, [d])

Convert the Python object `x` to a `T`.

If `d` is specified, it is returned on failure instead of throwing an error.
"""
pyconvert(::Type{T}, x) where {T} = @autopy x @pyconvert T x_ error(
    "cannot convert this Python '$(pytype(x_).__name__)' to a Julia '$T'",
)
pyconvert(::Type{T}, x, d) where {T} = @autopy x @pyconvert T x_ d

"""
    pyconvertarg(T, x, name)

Convert the Python object `x` to a `T`.

On failure, throws a Python `TypeError` saying that the argument `name` could not be converted.
"""
pyconvertarg(::Type{T}, x, name) where {T} = @autopy x @pyconvert T x_ begin
    errset(
        pybuiltins.TypeError,
        "Cannot convert argument '$name' to a Julia '$T', got a '$(pytype(x_).__name__)'",
    )
    pythrow()
end

function init_pyconvert()
    push!(PYCONVERT_EXTRATYPES, pyimport("io" => "IOBase"))
    push!(
        PYCONVERT_EXTRATYPES,
        pyimport("numbers" => ("Number", "Complex", "Real", "Rational", "Integral"))...,
    )
    push!(
        PYCONVERT_EXTRATYPES,
        pyimport("collections.abc" => ("Iterable", "Sequence", "Set", "Mapping"))...,
    )

    pyconvert_add_rule(pyconvert_rule_none, "builtins:NoneType", Nothing, Any)
    pyconvert_add_rule(pyconvert_rule_bool, "builtins:bool", Bool, Any)
    pyconvert_add_rule(pyconvert_rule_float, "builtins:float", Float64, Any)
    pyconvert_add_rule(
        pyconvert_rule_complex,
        "builtins:complex",
        Complex{Float64},
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_int, "numbers:Integral", Integer, Any)
    pyconvert_add_rule(pyconvert_rule_str, "builtins:str", String, Any)
    pyconvert_add_rule(
        pyconvert_rule_bytes,
        "builtins:bytes",
        Base.CodeUnits{UInt8,String},
        Any,
    )
    pyconvert_add_rule(
        pyconvert_rule_range,
        "builtins:range",
        StepRange{<:Integer,<:Integer},
        Any,
    )
    pyconvert_add_rule(
        pyconvert_rule_fraction,
        "numbers:Rational",
        Rational{<:Integer},
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_iterable, "builtins:tuple", NamedTuple, Any)
    pyconvert_add_rule(pyconvert_rule_iterable, "builtins:tuple", Tuple, Any)
    pyconvert_add_rule(pyconvert_rule_datetime, "datetime:datetime", DateTime, Any)
    pyconvert_add_rule(pyconvert_rule_date, "datetime:date", Date, Any)
    pyconvert_add_rule(pyconvert_rule_time, "datetime:time", Time, Any)
    pyconvert_add_rule(
        pyconvert_rule_timedelta,
        "datetime:timedelta",
        Microsecond,
        Any,
    )
    pyconvert_add_rule(
        pyconvert_rule_exception,
        "builtins:BaseException",
        PyException,
        Any,
    )

    pyconvert_add_rule(pyconvert_rule_none, "builtins:NoneType", Missing, Missing)
    pyconvert_add_rule(pyconvert_rule_bool, "builtins:bool", Number, Number)
    pyconvert_add_rule(pyconvert_rule_float, "numbers:Real", Number, Number)
    pyconvert_add_rule(pyconvert_rule_float, "builtins:float", Nothing, Nothing)
    pyconvert_add_rule(pyconvert_rule_float, "builtins:float", Missing, Missing)
    pyconvert_add_rule(pyconvert_rule_complex, "numbers:Complex", Number, Number)
    pyconvert_add_rule(pyconvert_rule_int, "numbers:Integral", Number, Number)
    pyconvert_add_rule(pyconvert_rule_str, "builtins:str", Symbol)
    pyconvert_add_rule(pyconvert_rule_str, "builtins:str", Char)
    pyconvert_add_rule(pyconvert_rule_bytes, "builtins:bytes", Vector{UInt8})
    pyconvert_add_rule(
        pyconvert_rule_range,
        "builtins:range",
        UnitRange{<:Integer},
    )
    pyconvert_add_rule(pyconvert_rule_fraction, "numbers:Rational", Number, Number)
    pyconvert_add_rule(
        pyconvert_rule_iterable,
        "collections.abc:Iterable",
        Vector,
    )
    pyconvert_add_rule(pyconvert_rule_iterable, "collections.abc:Iterable", Tuple)
    pyconvert_add_rule(pyconvert_rule_iterable, "collections.abc:Iterable", Pair)
    pyconvert_add_rule(pyconvert_rule_iterable, "collections.abc:Iterable", Set)
    pyconvert_add_rule(
        pyconvert_rule_iterable,
        "collections.abc:Sequence",
        Vector,
    )
    pyconvert_add_rule(pyconvert_rule_iterable, "collections.abc:Sequence", Tuple)
    pyconvert_add_rule(pyconvert_rule_iterable, "collections.abc:Set", Set)
    pyconvert_add_rule(pyconvert_rule_mapping, "collections.abc:Mapping", Dict)
    pyconvert_add_rule(
        pyconvert_rule_timedelta,
        "datetime:timedelta",
        Millisecond,
    )
    pyconvert_add_rule(pyconvert_rule_timedelta, "datetime:timedelta", Second)
    pyconvert_add_rule(pyconvert_rule_timedelta, "datetime:timedelta", Nanosecond)

    pyconvert_add_rule(pyconvert_rule_object, "builtins:object", Py, Any)
end
