struct PyConvertRule
    type :: Type
    func :: Function
    priority :: Int
end

const PYCONVERT_RULES = Dict{String, Vector{PyConvertRule}}()
const PYCONVERT_EXTRATYPES = Py[]

function pyconvert_add_rule(pytypename::String, type::Type, func::Function, priority::Int=0)
    @nospecialize type func
    push!(get!(Vector{PyConvertRule}, PYCONVERT_RULES, pytypename), PyConvertRule(type, func, priority))
    return
end

if false
    # this scheme returns either the result or Unconverted()
    struct Unconverted end
    pyconvert_return(x) = x
    pyconvert_unconverted() = Unconverted()
    pyconvert_returntype(::Type{T}) where {T} = Union{T,Unconverted}
    pyconvert_isunconverted(r) = r === Unconverted()
    pyconvert_result(::Type{T}, r) where {T} = r::T
elseif true
    # this scheme stores the result in PYCONVERT_RESULT
    const PYCONVERT_RESULT = Ref{Any}(nothing)
    pyconvert_return(x) = (PYCONVERT_RESULT[] = x; true)
    pyconvert_unconverted() = false
    pyconvert_returntype(::Type{T}) where {T} = Bool
    pyconvert_isunconverted(r::Bool) = !r
    pyconvert_result(::Type{T}, r::Bool) where {T} = (ans = PYCONVERT_RESULT[]::T; PYCONVERT_RESULT[] = nothing; ans)
else
    # same as the previous scheme, but with special handling for bits types
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
    pyconvert_unconverted() = false
    pyconvert_returntype(::Type{T}) where {T} = Bool
    pyconvert_isunconverted(r::Bool) = !r
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
    return "$m/$n"
end

function pyconvert_get_rules(type::Type, pytype::Py)
    @nospecialize type

    pyisin(x, ys) = any(pyis(x, y) for y in ys)

    # get the MROs of all base types we are considering
    omro = collect(pytype.__mro__)
    basetypes = Py[pytype]
    basemros = Vector{Py}[omro]
    for xtype in PYCONVERT_EXTRATYPES
        # find the topmost supertype of
        xbase = PyNULL
        for base in omro
            if pyissubclass(base, xtype)
                xbase = base
            end
        end
        if !pyisnull(xbase)
            push!(basetypes, xtype)
            xmro = collect(xtype.__mro__)
            pyisin(xbase, xmro) || pushfirst!(xmro, xbase)
            push!(basemros, xmro)
        end
    end
    for xbase in basetypes[2:end]
        push!(basemros, [xbase])
    end

    # merge the MROs
    # this is a port of the merge() function at the bottom of:
    # https://www.python.org/download/releases/2.3/mro/
    mro = Py[]
    while !isempty(basemros)
        # find the first head not contained in any tail
        ok = false
        b = PyNULL
        for bmro in basemros
            b = bmro[1]
            if all(bmro -> !pyisin(b, bmro[2:end]), basemros)
                ok = true
                break
            end
        end
        ok || error("Fatal inheritence error: could not merge MROs (mro=$mro, basemros=$basemros)")
        # add it to the list
        push!(mro, b)
        # remove it from consideration
        for bmro in basemros
            filter!(t -> !pyis(t, b), bmro)
        end
        # remove empty lists
        filter!(x -> !isempty(x), basemros)
    end
    # check the original MRO is preserved
    omro_ = filter(t -> pyisin(t, omro), mro)
    @assert length(omro) == length(omro_)
    @assert all(pyis(x,y) for (x,y) in zip(omro, omro_))

    # get the names of the types in the MRO of pytype
    xmro = [String[pyconvert_typename(t)] for t in mro]

    # add special names corresponding to certain interfaces
    # these get inserted just above the topmost type satisfying the interface
    for (t, x) in reverse(collect(zip(mro, xmro)))
        if pyhasattr(t, "__array_struct__")
            push!(x, "<arraystruct>")
            break
        end
    end
    for (t, x) in reverse(collect(zip(mro, xmro)))
        if pyhasattr(t, "__array_interface__")
            push!(x, "<arrayinterface>")
            break
        end
    end
    for (t, x) in reverse(collect(zip(mro, xmro)))
        if pyhasattr(t, "__array__")
            push!(x, "<array>")
            break
        end
    end
    for (t, x) in reverse(collect(zip(mro, xmro)))
        if C.PyType_CheckBuffer(getptr(t))
            push!(x, "<buffer>")
            break
        end
    end

    # flatten to get the MRO as a list of strings
    mro = String[x for xs in xmro for x in xs]

    # get corresponding rules
    rules = PyConvertRule[rule for tname in mro for rule in get!(Vector{PyConvertRule}, PYCONVERT_RULES, tname)]

    # order the rules by priority, then by original order
    order = sort(axes(rules, 1), by = i -> (-rules[i].priority, i))
    rules = rules[order]

    # TODO: everything up to here does not depend on the julia type and could be cached

    # intersect rules with type
    rules = PyConvertRule[PyConvertRule(typeintersect(rule.type, type), rule.func, rule.priority) for rule in rules]

    # explode out unions
    rules = [PyConvertRule(type, rule.func, rule.priority) for rule in rules for type in Utils.explode_union(rule.type)]

    # filter out empty rules
    rules = [rule for rule in rules if rule.type != Union{}]

    # filter out repeated rules
    rules = [rule for (i, rule) in enumerate(rules) if !any((rule.func === rules[j].func) && ((rule.type) <: (rules[j].type)) for j in 1:(i-1))]

    @debug "pyconvert" type pytype mro=join(mro, " ") rules
    return Function[pyconvert_fix(rule.type, rule.func) for rule in rules]
end

pyconvert_fix(::Type{T}, func) where {T} = x -> func(T, x)

const PYCONVERT_RULES_CACHE = Dict{C.PyPtr, Dict{Type, Vector{Function}}}()

function pyconvert_rule_fast(::Type{T}, x::Py) where {T}
    if T isa Union
        a = pyconvert_rule_fast(T.a, x) :: pyconvert_returntype(T.a)
        pyconvert_isunconverted(a) || return a
        b = pyconvert_rule_fast(T.b, x) :: pyconvert_returntype(T.b)
        pyconvert_isunconverted(b) || return b
    end
    pyconvert_unconverted()
end

pytryconvert(::Type{T}, x) where {T} = @autopy x begin
    # We can optimize the conversion for some types by overloading pytryconvert_fast.
    # It MUST give the same results as via the slower route using rules.
    ans = pyconvert_rule_fast(T, getpy(x_)) :: pyconvert_returntype(T)
    pyconvert_isunconverted(ans) || return ans
    # get rules from the cache
    # TODO: we should hold weak references and clear the cache if types get deleted
    tptr = C.Py_Type(getptr(x_))
    trules = get!(Dict{Type, Vector{PyConvertRule}}, PYCONVERT_RULES_CACHE, tptr)
    if !haskey(trules, T)
        t = pynew(incref(tptr))
        trules[T] = pyconvert_get_rules(T, t)
        pydel!(t)
    end
    rules = trules[T]
    # apply the rules
    for rule in rules
        ans = rule(getpy(x_)) :: pyconvert_returntype(T)
        pyconvert_isunconverted(ans) || return ans
    end
    return pyconvert_unconverted()
end

"""
    @pyconvert(T, x, [onfail])

Convert the Python object `x` to a `T`.

On failure, evaluates to `onfail`, which defaults to `return pyconvert_unconverted()` (mainly useful for writing conversion rules).
"""
macro pyconvert(T, x, onfail=:(return $pyconvert_unconverted()))
    quote
        ans = pytryconvert($(esc(T)), $(esc(x)))
        if pyconvert_isunconverted(ans)
            $(esc(onfail))
        else
            pyconvert_result($(esc(T)), ans)
        end
    end
end
export @pyconvert

"""
    pyconvert(T, x, [d])

Convert the Python object `x` to a `T`.

If `d` is specified, it is returned on failure instead of throwing an error.
"""
pyconvert(::Type{T}, x) where {T} = @autopy x @pyconvert T x_ error("cannot convert this Python '$(pytype(x_).__name__)' to a Julia '$T'")
pyconvert(::Type{T}, x, d) where {T} = @autopy x @pyconvert T x_ d
export pyconvert

"""
    pyconvertarg(T, x, name)

Convert the Python object `x` to a `T`.

On failure, throws a Python `TypeError` saying that the argument `name` could not be converted.
"""
pyconvertarg(::Type{T}, x, name) where {T} = @autopy x @pyconvert T x_ begin
    errset(pybuiltins.TypeError, "Cannot convert argument '$name' to a Julia '$T', got a '$(pytype(x_).__name__)'")
    pythrow()
end

macro pyconvert_and_del(T, x, onfail=:(return $pyconvert_unconverted()))
    quote
        ans = pytryconvert($(esc(T)), $(esc(x)))
        pydel!($(esc(x)))
        if pyconvert_isunconverted(ans)
            $(esc(onfail))
        else
            pyconvert_result($(esc(T)), ans)
        end
    end
end

pyconvert_and_del(::Type{T}, x) where {T} = begin
    ans = pyconvert(T, x)
    pydel!(x)
    ans
end

function init_pyconvert()
    push!(PYCONVERT_EXTRATYPES, pyimport("io"=>"IOBase"))
    push!(PYCONVERT_EXTRATYPES, pyimport("numbers"=>("Number", "Complex", "Real", "Rational", "Integral"))...)
    push!(PYCONVERT_EXTRATYPES, pyimport("collections.abc" => ("Iterable", "Sequence", "Set", "Mapping"))...)

    # priority 300: wrapped julia values
    pyconvert_add_rule("juliacall/As", Any, pyconvert_rule_jlas, 300)
    pyconvert_add_rule("juliacall/ValueBase", Any, pyconvert_rule_jlvalue, 300)
    # priority 200: arrays
    pyconvert_add_rule("<arraystruct>", PyArray, pyconvert_rule_array_nocopy, 200)
    pyconvert_add_rule("<arrayinterface>", PyArray, pyconvert_rule_array_nocopy, 200)
    pyconvert_add_rule("<array>", PyArray, pyconvert_rule_array_nocopy, 200)
    pyconvert_add_rule("<buffer>", PyArray, pyconvert_rule_array_nocopy, 200)
    # priority 100: canonical
    pyconvert_add_rule("builtins/NoneType", Nothing, pyconvert_rule_none, 100)
    pyconvert_add_rule("builtins/bool", Bool, pyconvert_rule_bool, 100)
    pyconvert_add_rule("builtins/float", Float64, pyconvert_rule_float, 100)
    pyconvert_add_rule("builtins/complex", Complex{Float64}, pyconvert_rule_complex, 100)
    pyconvert_add_rule("numbers/Integral", Integer, pyconvert_rule_int, 100)
    pyconvert_add_rule("builtins/str", String, pyconvert_rule_str, 100)
    pyconvert_add_rule("builtins/bytes", Base.CodeUnits{UInt8,String}, pyconvert_rule_bytes, 100)
    pyconvert_add_rule("builtins/range", StepRange{<:Integer,<:Integer}, pyconvert_rule_range, 100)
    pyconvert_add_rule("numbers/Rational", Rational{<:Integer}, pyconvert_rule_fraction, 100)
    pyconvert_add_rule("collections.abc/Iterable", PyIterable, pyconvert_rule_iterable, 100)
    pyconvert_add_rule("collections.abc/Sequence", PyList, pyconvert_rule_sequence, 100)
    pyconvert_add_rule("collections.abc/Set", PySet, pyconvert_rule_set, 100)
    pyconvert_add_rule("collections.abc/Mapping", PyDict, pyconvert_rule_mapping, 100)
    pyconvert_add_rule("io/IOBase", PyIO, pyconvert_rule_io, 100)
    pyconvert_add_rule("_io/_IOBase", PyIO, pyconvert_rule_io, 100)
    pyconvert_add_rule("pandas.core.frame/DataFrame", PyPandasDataFrame, pyconvert_rule_pandasdataframe, 100)
    pyconvert_add_rule("builtins/tuple", Tuple, pyconvert_rule_iterable, 100)
    pyconvert_add_rule("datetime/datetime", DateTime, pyconvert_rule_datetime, 100)
    pyconvert_add_rule("datetime/date", Date, pyconvert_rule_date, 100)
    pyconvert_add_rule("datetime/time", Time, pyconvert_rule_time, 100)
    pyconvert_add_rule("builtins/BaseException", PyException, pyconvert_rule_exception, 100)
    # priority 0: reasonable
    pyconvert_add_rule("builtins/NoneType", Missing, pyconvert_rule_none)
    pyconvert_add_rule("builtins/bool", Number, pyconvert_rule_bool)
    pyconvert_add_rule("numbers/Real", Number, pyconvert_rule_float)
    pyconvert_add_rule("builtins/float", Nothing, pyconvert_rule_float)
    pyconvert_add_rule("builtins/float", Missing, pyconvert_rule_float)
    pyconvert_add_rule("numbers/Complex", Number, pyconvert_rule_complex)
    pyconvert_add_rule("numbers/Integral", Number, pyconvert_rule_int)
    pyconvert_add_rule("builtins/str", Symbol, pyconvert_rule_str)
    pyconvert_add_rule("builtins/str", Char, pyconvert_rule_str)
    pyconvert_add_rule("builtins/bytes", Vector{UInt8}, pyconvert_rule_bytes)
    pyconvert_add_rule("builtins/range", UnitRange{<:Integer}, pyconvert_rule_range)
    pyconvert_add_rule("numbers/Rational", Number, pyconvert_rule_fraction)
    pyconvert_add_rule("collections.abc/Iterable", Vector, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Iterable", Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Iterable", Pair, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Iterable", Set, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Sequence", Vector, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Sequence", Tuple, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Set", Set, pyconvert_rule_iterable)
    pyconvert_add_rule("collections.abc/Mapping", Dict, pyconvert_rule_mapping)
    pyconvert_add_rule("<arraystruct>", Array, pyconvert_rule_array)
    pyconvert_add_rule("<arrayinterface>", Array, pyconvert_rule_array)
    pyconvert_add_rule("<array>", Array, pyconvert_rule_array)
    pyconvert_add_rule("<buffer>", Array, pyconvert_rule_array)
    pyconvert_add_rule("<arraystruct>", AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<arrayinterface>", AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<array>", AbstractArray, pyconvert_rule_array)
    pyconvert_add_rule("<buffer>", AbstractArray, pyconvert_rule_array)
    # priority -100: fallbacks
    pyconvert_add_rule("builtins/object", Py, pyconvert_rule_object, -100)
    # priority -200: explicit
end
