struct PyConvertRule
    pytypename::String
    type::Type
    scope::Type
    func::Function
    order::Int
end

const PYCONVERT_RULES = PyConvertRule[]

"""
    pyconvert_add_rule(tname::String, T::Type, S::Type, func::Function)

Add a new conversion rule for `pyconvert`.

### Arguments

- `tname` is a string of the form `"__module__:__qualname__"` identifying a Python type `t`,
  such as `"builtins:dict"` or `"sympy:Symbol"`. This rule only applies to
  Python objects of this type.
- `T` is a Julia type, such that this rule only applies when the target type intersects
  with `T`.
- `S` is the scope of the rule and must be a supertype of `T`. The requested target type
  must be a subtype of `S` (for a union, at least one member must be a subtype).
- `func` is the function implementing the rule.

When `pyconvert(R, x)` is called, rules for which `typeintersect(T, R) != Union{}`,
`R <: S`, and `pyisinstance(x, t)` are considered. Rules added later are tried first.

### Implementing `func`

`func` is called as `func(U, x::Py)` for some `U <: T`.

It must return one of:
- `pyconvert_return(ans)` where `ans` is the result of the conversion (and must be a `U`).
- `pyconvert_unconverted()` if the conversion was not possible (e.g. converting a `list` to
  `Vector{Int}` might fail if some of the list items are not integers).

The target type `U` is never a union or the empty type, i.e. it is always a data type or
union-all.

"""
function pyconvert_add_rule(
    pytypename::String,
    type::Type,
    scope::Type,
    func::Function,
)
    @nospecialize type scope func
    type <: scope || throw(
        ArgumentError("conversion rule target $type is not a subtype of its scope $scope"),
    )
    push!(
        PYCONVERT_RULES,
        PyConvertRule(pytypename, type, scope, func, length(PYCONVERT_RULES)),
    )
    empty!.(values(PYCONVERT_RULES_CACHE))
    return
end

function pyconvert_add_rule_high_priority(
    pytypename::String,
    type::Type,
    scope::Type,
    func::Function,
)
    @nospecialize type scope func
    type <: scope || throw(
        ArgumentError("conversion rule target $type is not a subtype of its scope $scope"),
    )
    push!(
        PYCONVERT_RULES,
        PyConvertRule(
            pytypename,
            type,
            scope,
            func,
            typemax(Int) - length(PYCONVERT_RULES),
        ),
    )
    empty!.(values(PYCONVERT_RULES_CACHE))
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

function pyconvert_issubclass(pytype::Py, pytypename::String)
    pytypename == "<arraystruct>" && return pyhasattr(pytype, "__array_struct__")
    pytypename == "<arrayinterface>" && return pyhasattr(pytype, "__array_interface__")
    pytypename == "<array>" && return pyhasattr(pytype, "__array__")
    pytypename == "<buffer>" && return C.PyType_CheckBuffer(pytype)

    modname, typename = split(pytypename, ':'; limit = 2)
    modules = pyimport("sys").modules
    pyhasitem(modules, modname) || return false
    type = pygetitem(modules, modname)
    for name in split(typename, '.')
        pyhasattr(type, name) || return false
        type = pygetattr(type, name)
    end
    return pyissubclass(pytype, type)
end

function _pyconvert_get_rules(pytype::Py)
    rules = PyConvertRule[
        rule for rule in PYCONVERT_RULES if pyconvert_issubclass(pytype, rule.pytypename)
    ]

    sort!(rules; by = rule -> rule.order, rev = true)

    @debug "pyconvert" pytype rules
    return rules
end

const PYCONVERT_PREFERRED_TYPE = Dict{Py,Type}()

pyconvert_preferred_type(pytype::Py) =
    get!(PYCONVERT_PREFERRED_TYPE, pytype) do
        if pyissubclass(pytype, pybuiltins.int)
            Union{Int,BigInt}
        else
            _pyconvert_get_rules(pytype)[1].type
        end
    end

function pyconvert_get_rules(type::Type, pytype::Py)
    @nospecialize type

    # this could be cached
    rules = _pyconvert_get_rules(pytype)

    # A union is in scope when at least one of its components is in scope.
    rules = [rule for rule in rules if any(t -> t <: rule.scope, Utils.explode_union(type))]

    # intersect rules with type
    rules = PyConvertRule[
        PyConvertRule(
            rule.pytypename,
            typeintersect(rule.type, type),
            rule.scope,
            rule.func,
            rule.order,
        ) for
        rule in rules
    ]

    # explode out unions
    rules = [
        PyConvertRule(rule.pytypename, type, rule.scope, rule.func, rule.order) for
        rule in rules for type in Utils.explode_union(rule.type)
    ]

    # filter out empty rules
    rules = [rule for rule in rules if rule.type != Union{}]

    # filter out repeated rules
    rules = [
        rule for (i, rule) in enumerate(rules) if !any(
            (rule.func === rules[j].func) && ((rule.type) <: (rules[j].type)) for
            j = 1:(i-1)
        )
    ]

    @debug "pyconvert" type rules
    return Function[pyconvert_fix(rule.type, rule.func) for rule in rules]
end

pyconvert_fix(::Type{T}, func) where {T} = x -> func(T, x)

const PYCONVERT_RULES_CACHE = Dict{Type,Dict{C.PyPtr,Vector{Function}}}()

@generated pyconvert_rules_cache(::Type{T}) where {T} =
    get!(Dict{C.PyPtr,Vector{Function}}, PYCONVERT_RULES_CACHE, T)

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
        unsafe_pydel(t)
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
    pyimport("io")
    pyimport("types")
    pyimport("numbers")
    pyimport("collections.abc")

    pyconvert_add_rule("builtins:object", Py, Any, pyconvert_rule_object)

    pyconvert_add_rule("types:NoneType", Missing, Missing, pyconvert_rule_none)
    pyconvert_add_rule("builtins:bool", Number, Number, pyconvert_rule_bool)
    pyconvert_add_rule("numbers:Rational", Number, Number, pyconvert_rule_fraction)
    pyconvert_add_rule("numbers:Real", Number, Number, pyconvert_rule_float)
    pyconvert_add_rule("builtins:float", Nothing, Nothing, pyconvert_rule_float)
    pyconvert_add_rule("builtins:float", Missing, Missing, pyconvert_rule_float)
    pyconvert_add_rule("numbers:Complex", Number, Number, pyconvert_rule_complex)
    pyconvert_add_rule("numbers:Integral", Number, Number, pyconvert_rule_int)
    pyconvert_add_rule("builtins:str", Symbol, Symbol, pyconvert_rule_str)
    pyconvert_add_rule("builtins:str", Char, Char, pyconvert_rule_str)
    pyconvert_add_rule("builtins:bytes", Vector{UInt8}, Vector{UInt8}, pyconvert_rule_bytes)
    pyconvert_add_rule(
        "builtins:range",
        UnitRange{<:Integer},
        UnitRange{<:Integer},
        pyconvert_rule_range,
    )
    pyconvert_add_rule(
        "collections.abc:Iterable",
        Vector,
        AbstractArray,
        pyconvert_rule_iterable,
    )
    pyconvert_add_rule("collections.abc:Iterable", Tuple, Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Iterable", Pair, Pair, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Iterable", Set, Set, pyconvert_rule_iterable)
    pyconvert_add_rule(
        "collections.abc:Sequence",
        Vector,
        AbstractArray,
        pyconvert_rule_iterable,
    )
    pyconvert_add_rule("collections.abc:Sequence", Tuple, Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Set", Set, Set, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc:Mapping", Dict, Dict, pyconvert_rule_mapping)
    pyconvert_add_rule(
        "datetime:timedelta",
        Millisecond,
        Millisecond,
        pyconvert_rule_timedelta,
    )
    pyconvert_add_rule("datetime:timedelta", Second, Second, pyconvert_rule_timedelta)
    pyconvert_add_rule("datetime:timedelta", Nanosecond, Nanosecond, pyconvert_rule_timedelta)
end

function init_pyconvert_canonical()
    pyconvert_add_rule("types:NoneType", Nothing, Any, pyconvert_rule_none)
    pyconvert_add_rule("builtins:bool", Bool, Any, pyconvert_rule_bool)
    pyconvert_add_rule("builtins:float", Float64, Any, pyconvert_rule_float)
    pyconvert_add_rule(
        "builtins:complex",
        Complex{Float64},
        Any,
        pyconvert_rule_complex,
    )
    pyconvert_add_rule(
        "numbers:Rational",
        Rational{<:Integer},
        Any,
        pyconvert_rule_fraction,
    )
    pyconvert_add_rule("numbers:Integral", Integer, Any, pyconvert_rule_int)
    pyconvert_add_rule("builtins:str", String, Any, pyconvert_rule_str)
    pyconvert_add_rule(
        "builtins:bytes",
        Base.CodeUnits{UInt8,String},
        Any,
        pyconvert_rule_bytes,
    )
    pyconvert_add_rule(
        "builtins:range",
        StepRange{<:Integer,<:Integer},
        Any,
        pyconvert_rule_range,
    )
    pyconvert_add_rule("builtins:tuple", NamedTuple, Any, pyconvert_rule_iterable)
    pyconvert_add_rule("builtins:tuple", Tuple, Any, pyconvert_rule_iterable)
    pyconvert_add_rule("datetime:datetime", DateTime, Any, pyconvert_rule_datetime)
    pyconvert_add_rule("datetime:date", Date, Any, pyconvert_rule_date)
    pyconvert_add_rule("datetime:time", Time, Any, pyconvert_rule_time)
    pyconvert_add_rule(
        "datetime:timedelta",
        Microsecond,
        Any,
        pyconvert_rule_timedelta,
    )
    pyconvert_add_rule(
        "builtins:BaseException",
        PyException,
        Any,
        pyconvert_rule_exception,
    )
end
