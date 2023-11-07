### object

pyconvert_rule_object(::Type{Py}, x::Py) = pyconvert_return(x)

### Exception

pyconvert_rule_exception(::Type{R}, x::Py) where {R<:PyException} = pyconvert_return(PyException(x))

### None

pyconvert_rule_none(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_none(::Type{Missing}, x::Py) = pyconvert_return(missing)

### Bool

function pyconvert_rule_bool(::Type{T}, x::Py) where {T<:Number}
    val = pybool_asbool(x)
    if T in (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, BigInt)
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end

### str

pyconvert_rule_str(::Type{String}, x::Py) = pyconvert_return(pystr_asstring(x))
pyconvert_rule_str(::Type{Symbol}, x::Py) = pyconvert_return(Symbol(pystr_asstring(x)))
pyconvert_rule_str(::Type{Char}, x::Py) = begin
    s = pystr_asstring(x)
    if length(s) == 1
        pyconvert_return(first(s))
    else
        pyconvert_unconverted()
    end
end

### bytes

pyconvert_rule_bytes(::Type{Vector{UInt8}}, x::Py) = pyconvert_return(copy(pybytes_asvector(x)))
pyconvert_rule_bytes(::Type{Base.CodeUnits{UInt8,String}}, x::Py) = pyconvert_return(codeunits(pybytes_asUTF8string(x)))

### int

pyconvert_rule_int(::Type{T}, x::Py) where {T<:Number} = begin
    # first try to convert to Clonglong (or Culonglong if unsigned)
    v = T <: Unsigned ? C.PyLong_AsUnsignedLongLong(x) : C.PyLong_AsLongLong(x)
    if !iserrset_ambig(v)
        # success
        return pyconvert_tryconvert(T, v)
    elseif errmatches(pybuiltins.OverflowError)
        # overflows Clonglong or Culonglong
        errclear()
        if T in (
               Bool,
               Int8,
               Int16,
               Int32,
               Int64,
               Int128,
               UInt8,
               UInt16,
               UInt32,
               UInt64,
               UInt128,
           ) &&
           typemin(typeof(v)) ≤ typemin(T) &&
           typemax(T) ≤ typemax(typeof(v))
            # definitely overflows S, give up now
            return pyconvert_unconverted()
        else
            # try converting -> int -> str -> BigInt -> T
            x_int = pyint(x)
            x_str = pystr(String, x_int)
            pydel!(x_int)
            v = parse(BigInt, x_str)
            return pyconvert_tryconvert(T, v)
        end
    else
        # other error
        pythrow()
    end
end

### float

function pyconvert_rule_float(::Type{T}, x::Py) where {T<:Number}
    val = pyfloat_asdouble(x)
    if T in (Float16, Float32, Float64, BigFloat)
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end

# NaN is sometimes used to represent missing data of other types
# so we allow converting it to Nothing or Missing
function pyconvert_rule_float(::Type{Nothing}, x::Py)
    val = pyfloat_asdouble(x)
    if isnan(val)
        pyconvert_return(nothing)
    else
        pyconvert_unconverted()
    end
end

function pyconvert_rule_float(::Type{Missing}, x::Py)
    val = pyfloat_asdouble(x)
    if isnan(val)
        pyconvert_return(missing)
    else
        pyconvert_unconverted()
    end
end

### complex

function pyconvert_rule_complex(::Type{T}, x::Py) where {T<:Number}
    val = pycomplex_ascomplex(x)
    if T in (Complex{Float64}, Complex{Float32}, Complex{Float16}, Complex{BigFloat})
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end

### range

function pyconvert_rule_range(::Type{R}, x::Py, ::Type{StepRange{T0,S0}}=Utils._type_lb(R), ::Type{StepRange{T1,S1}}=Utils._type_ub(R)) where {R<:StepRange,T0,S0,T1,S1}
    a = @pyconvert(Utils._typeintersect(Integer, T1), x.start)
    b = @pyconvert(Utils._typeintersect(Integer, S1), x.step)
    c = @pyconvert(Utils._typeintersect(Integer, T1), x.stop)
    a′, c′ = promote(a, c - oftype(c, sign(b)))
    T2 = Utils._promote_type_bounded(T0, typeof(a′), typeof(c′), T1)
    S2 = Utils._promote_type_bounded(S0, typeof(c′), S1)
    pyconvert_return(StepRange{T2, S2}(a′, b, c′))
end

function pyconvert_rule_range(::Type{R}, x::Py, ::Type{UnitRange{T0}}=Utils._type_lb(R), ::Type{UnitRange{T1}}=Utils._type_ub(R)) where {R<:UnitRange,T0,T1}
    b = @pyconvert(Int, x.step)
    b == 1 || return pyconvert_unconverted()
    a = @pyconvert(Utils._typeintersect(Integer, T1), x.start)
    c = @pyconvert(Utils._typeintersect(Integer, T1), x.stop)
    a′, c′ = promote(a, c - oftype(c, 1))
    T2 = Utils._promote_type_bounded(T0, typeof(a′), typeof(c′), T1)
    pyconvert_return(UnitRange{T2}(a′, c′))
end

### fraction

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{R}, x::Py, ::Type{Rational{T0}}=Utils._type_lb(R), ::Type{Rational{T1}}=Utils._type_ub(R)) where {R<:Rational,T0,T1}
    a = @pyconvert(Utils._typeintersect(Integer, T1), x.numerator)
    b = @pyconvert(Utils._typeintersect(Integer, T1), x.denominator)
    a, b = promote(a, b)
    T2 = Utils._promote_type_bounded(T0, typeof(a), typeof(b), T1)
    pyconvert_return(Rational{T2}(a, b))
end

# works for any collections.abc.Rational
function pyconvert_rule_fraction(::Type{T}, x::Py) where {T<:Number}
    pyconvert_tryconvert(T, @pyconvert(Rational{<:Integer}, x))
end

### collections

# Vector

function _pyconvert_rule_iterable(ans::Vector{T0}, it::Py, ::Type{T1}) where {T0,T1}
    @label again
    x_ = unsafe_pynext(it)
    if pyisnull(x_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    x = @pyconvert(T1, x_)
    if x isa T0
        push!(ans, x)
        @goto again
    end
    T2 = Utils._promote_type_bounded(T0, typeof(x), T1)
    ans2 = Vector{T2}(ans)
    push!(ans2, x)
    return _pyconvert_rule_iterable(ans2, it, T1)
end

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Vector{T0}}=Utils._type_lb(R), ::Type{Vector{T1}}=Utils._type_ub(R)) where {R<:Vector,T0,T1}
    it = pyiter(x)
    ans = Vector{T0}()
    return _pyconvert_rule_iterable(ans, it, T1)
end

# Set

function _pyconvert_rule_iterable(ans::Set{T0}, it::Py, ::Type{T1}) where {T0,T1}
    @label again
    x_ = unsafe_pynext(it)
    if pyisnull(x_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    x = @pyconvert(T1, x_)
    if x isa T0
        push!(ans, x)
        @goto again
    end
    T2 = Utils._promote_type_bounded(T0, typeof(x), T1)
    ans2 = Set{T2}(ans)
    push!(ans2, x)
    return _pyconvert_rule_iterable(ans2, it, T1)
end

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Set{T0}}=Utils._type_lb(R), ::Type{Set{T1}}=Utils._type_ub(R)) where {R<:Set,T0,T1}
    it = pyiter(x)
    ans = Set{T0}()
    return _pyconvert_rule_iterable(ans, it, T1)
end

# Dict

function _pyconvert_rule_mapping(ans::Dict{K0,V0}, x::Py, it::Py, ::Type{K1}, ::Type{V1}) where {K0,V0,K1,V1}
    @label again
    k_ = unsafe_pynext(it)
    if pyisnull(k_)
        pydel!(it)
        return pyconvert_return(ans)
    end
    v_ = pygetitem(x, k_)
    k = @pyconvert(K1, k_)
    v = @pyconvert(V1, v_)
    if k isa K0 && v isa V0
        push!(ans, k => v)
        @goto again
    end
    K2 = Utils._promote_type_bounded(K0, typeof(k), K1)
    V2 = Utils._promote_type_bounded(V0, typeof(v), V1)
    ans2 = Dict{K2,V2}(ans)
    push!(ans2, k => v)
    return _pyconvert_rule_mapping(ans2, x, it, K1, V1)
end

function pyconvert_rule_mapping(::Type{R}, x::Py, ::Type{Dict{K0,V0}}=Utils._type_lb(R), ::Type{Dict{K1,V1}}=Utils._type_ub(R)) where {R<:Dict,K0,V0,K1,V1}
    it = pyiter(x)
    ans = Dict{K0,V0}()
    return _pyconvert_rule_mapping(ans, x, it, K1, V1)
end

# Tuple

function pyconvert_rule_iterable(::Type{T}, xs::Py) where {T<:Tuple}
    T isa DataType || return pyconvert_unconverted()
    if T != Tuple{} && Tuple{T.parameters[end]} == Base.tuple_type_tail(Tuple{T.parameters[end]})
        isvararg = true
        vartype = Base.tuple_type_head(Tuple{T.parameters[end]})
        ts = T.parameters[1:end-1]
    else
        isvararg = false
        vartype = Union{}
        ts = T.parameters
    end
    zs = Any[]
    for x in xs
        if length(zs) < length(ts)
            t = ts[length(zs) + 1]
        elseif isvararg
            t = vartype
        else
            return pyconvert_unconverted()
        end
        z = @pyconvert(t, x)
        push!(zs, z)
    end
    return length(zs) < length(ts) ? pyconvert_unconverted() : pyconvert_return(T(zs))
end

for N in 0:16
    Ts = [Symbol("T", n) for n in 1:N]
    zs = [Symbol("z", n) for n in 1:N]
    # Tuple with N elements
    @eval function pyconvert_rule_iterable(::Type{Tuple{$(Ts...)}}, xs::Py) where {$(Ts...)}
        xs = pytuple(xs)
        n = pylen(xs)
        n == $N || return pyconvert_unconverted()
        $((
            :($z = @pyconvert($T, pytuple_getitem(xs, $(i-1))))
            for (i, T, z) in zip(1:N, Ts, zs)
        )...)
        return pyconvert_return(($(zs...),))
    end
    # Tuple with N elements plus Vararg
    @eval function pyconvert_rule_iterable(::Type{Tuple{$(Ts...),Vararg{V}}}, xs::Py) where {$(Ts...),V}
        xs = pytuple(xs)
        n = pylen(xs)
        n ≥ $N || return pyconvert_unconverted()
        $((
            :($z = @pyconvert($T, pytuple_getitem(xs, $(i-1))))
            for (i, T, z) in zip(1:N, Ts, zs)
        )...)
        vs = V[]
        for i in $(N+1):n
            v = @pyconvert(V, pytuple_getitem(xs, i-1))
            push!(vs, v)
        end
        return pyconvert_return(($(zs...), vs...))
    end
end

# Pair

function pyconvert_rule_iterable(::Type{R}, x::Py, ::Type{Pair{K0,V0}}=Utils._type_lb(R), ::Type{Pair{K1,V1}}=Utils._type_ub(R)) where {R<:Pair,K0,V0,K1,V1}
    it = pyiter(x)
    k_ = unsafe_pynext(it)
    if pyisnull(k_)
        pydel!(it)
        pydel!(k_)
        return pyconvert_unconverted()
    end
    k = @pyconvert(K1, k_)
    v_ = unsafe_pynext(it)
    if pyisnull(v_)
        pydel!(it)
        pydel!(v_)
        return pyconvert_unconverted()
    end
    v = @pyconvert(V1, v_)
    z_ = unsafe_pynext(it)
    pydel!(it)
    if pyisnull(z_)
        pydel!(z_)
    else
        pydel!(z_)
        return pyconvert_unconverted()
    end
    K2 = Utils._promote_type_bounded(K0, typeof(k), K1)
    V2 = Utils._promote_type_bounded(V0, typeof(v), V1)
    return pyconvert_return(Pair{K2,V2}(k, v))
end

# NamedTuple

_nt_names_types(::Type) = nothing
_nt_names_types(::Type{NamedTuple}) = (nothing, nothing)
_nt_names_types(::Type{NamedTuple{names}}) where {names} = (names, nothing)
_nt_names_types(::Type{NamedTuple{names,types} where {names}}) where {types} = (nothing, types)
_nt_names_types(::Type{NamedTuple{names,types}}) where {names,types} = (names, types)

function pyconvert_rule_iterable(::Type{R}, x::Py) where {R<:NamedTuple}
    # this is actually strict and only converts python named tuples (i.e. tuples with a
    # _fields attribute) where the field names match those from R (if specified).
    names_types = _nt_names_types(R)
    names_types === nothing && return pyconvert_unconverted()
    names, types = names_types
    pyistuple(x) || return pyconvert_unconverted()
    names2_ = pygetattr(x, "_fields", pybuiltins.None)
    names2 = @pyconvert(names === nothing ? Tuple{Vararg{Symbol}} : typeof(names), names2_)
    pydel!(names2_)
    names === nothing || names === names2 || return pyconvert_unconverted()
    types2 = types === nothing ? NTuple{length(names2),Any} : types
    vals = @pyconvert(types2, x)
    length(vals) == length(names2) || return pyconvert_unconverted()
    types3 = types === nothing ? typeof(vals) : types
    return pyconvert_return(NamedTuple{names2,types3}(vals))
end

### datetime


function pyconvert_rule_date(::Type{Date}, x::Py)
    # datetime is a subtype of date, but we shouldn't convert datetime to Date since it's lossy
    pyisinstance(x, pydatetimetype) && return pyconvert_unconverted()
    year = pyconvert(Int, x.year)
    month = pyconvert(Int, x.month)
    day = pyconvert(Int, x.day)
    pyconvert_return(Date(year, month, day))
end

function pyconvert_rule_time(::Type{Time}, x::Py)
    pytime_isaware(x) && return pyconvert_unconverted()
    hour = pyconvert(Int, x.hour)
    minute = pyconvert(Int, x.minute)
    second = pyconvert(Int, x.second)
    microsecond = pyconvert(Int, x.microsecond)
    return pyconvert_return(Time(hour, minute, second, div(microsecond, 1000), mod(microsecond, 1000)))
end

function pyconvert_rule_datetime(::Type{DateTime}, x::Py)
    pydatetime_isaware(x) && return pyconvert_unconverted()
    # compute the time since _base_datetime
    # this accounts for fold
    d = x - _base_pydatetime
    days = pyconvert(Int, d.days)
    seconds = pyconvert(Int, d.seconds)
    microseconds = pyconvert(Int, d.microseconds)
    pydel!(d)
    iszero(mod(microseconds, 1000)) || return pyconvert_unconverted()
    return pyconvert_return(_base_datetime + Millisecond(div(microseconds, 1000) + 1000 * (seconds + 60 * 60 * 24 * days)))
end
