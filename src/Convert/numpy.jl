struct pyconvert_rule_numpysimplevalue{R,S} <: Function end

function (::pyconvert_rule_numpysimplevalue{R,SAFE})(::Type{T}, x::Py) where {R,SAFE,T}
    ans = C.PySimpleObject_GetValue(R, x)
    if SAFE
        pyconvert_return(convert(T, ans))
    else
        pyconvert_tryconvert(T, ans)
    end
end

function pyconvert_rule_datetime64(::Type{DateTime64}, x::Py)
    pyconvert_return(C.PySimpleObject_GetValue(DateTime64, x))
end

function pyconvert_rule_datetime64(::Type{T}, x::Py) where {T<:InlineDateTime64}
    pyconvert_tryconvert(T, C.PySimpleObject_GetValue(DateTime64, x))
end

function pyconvert_rule_datetime64(::Type{T}, x::Py) where {T<:NumpyDates.DatesInstant}
    d = C.PySimpleObject_GetValue(DateTime64, x)
    if isnan(d)
        pyconvert_unconverted()
    else
        pyconvert_tryconvert(T, d)
    end
end

function pyconvert_rule_datetime64(::Type{Missing}, x::Py)
    d = C.PySimpleObject_GetValue(DateTime64, x)
    if isnan(d)
        pyconvert_return(missing)
    else
        pyconvert_unconverted()
    end
end

function pyconvert_rule_datetime64(::Type{Nothing}, x::Py)
    d = C.PySimpleObject_GetValue(DateTime64, x)
    if isnan(d)
        pyconvert_return(nothing)
    else
        pyconvert_unconverted()
    end
end

function pyconvert_rule_timedelta64(::Type{TimeDelta64}, x::Py)
    pyconvert_return(C.PySimpleObject_GetValue(TimeDelta64, x))
end

function pyconvert_rule_timedelta64(::Type{T}, x::Py) where {T<:InlineTimeDelta64}
    pyconvert_tryconvert(T, C.PySimpleObject_GetValue(TimeDelta64, x))
end

function pyconvert_rule_timedelta64(::Type{T}, x::Py) where {T<:NumpyDates.DatesPeriod}
    d = C.PySimpleObject_GetValue(TimeDelta64, x)
    if isnan(d)
        pyconvert_unconverted()
    else
        pyconvert_tryconvert(T, d)
    end
end

function pyconvert_rule_timedelta64(::Type{Missing}, x::Py)
    d = C.PySimpleObject_GetValue(TimeDelta64, x)
    if isnan(d)
        pyconvert_return(missing)
    else
        pyconvert_unconverted()
    end
end

function pyconvert_rule_timedelta64(::Type{Nothing}, x::Py)
    d = C.PySimpleObject_GetValue(TimeDelta64, x)
    if isnan(d)
        pyconvert_return(missing)
    else
        pyconvert_unconverted()
    end
end

const NUMPY_SIMPLE_TYPES = [
    ("bool_", Bool),
    ("int8", Int8),
    ("int16", Int16),
    ("int32", Int32),
    ("int64", Int64),
    ("uint8", UInt8),
    ("uint16", UInt16),
    ("uint32", UInt32),
    ("uint64", UInt64),
    ("float16", Float16),
    ("float32", Float32),
    ("float64", Float64),
    ("complex32", ComplexF16),
    ("complex64", ComplexF32),
    ("complex128", ComplexF64),
]

function numpy_rule_specs()
    specs = PyConvertRuleSpec[]
    # simple numeric scalar types
    for (t, T) in NUMPY_SIMPLE_TYPES
        isbool = occursin("bool", t)
        isint = occursin("int", t) || isbool
        isuint = occursin("uint", t) || isbool
        isfloat = occursin("float", t)
        iscomplex = occursin("complex", t)
        isreal = isint || isfloat
        isnumber = isreal || iscomplex

        name = "numpy:$t"
        rule = pyconvert_rule_numpysimplevalue{T,false}()
        saferule = pyconvert_rule_numpysimplevalue{T,true}()

        push!(specs, (func = saferule, tname = name, type = T, scope = Any))
        isuint && push!(
            specs,
            (
                func = sizeof(T) ≤ sizeof(UInt) ? saferule : rule,
                tname = name,
                type = UInt,
                scope = UInt,
            ),
        )
        isuint && push!(
            specs,
            (
                func = sizeof(T) < sizeof(Int) ? saferule : rule,
                tname = name,
                type = Int,
                scope = Int,
            ),
        )
        isint && !isuint && push!(
            specs,
            (func = sizeof(T) ≤ sizeof(Int) ? saferule : rule, tname = name, type = Int, scope = Int),
        )
        isint && push!(specs, (func = rule, tname = name, type = Integer, scope = Integer))
        isfloat && push!(specs, (func = saferule, tname = name, type = Float64, scope = Float64))
        isreal && push!(specs, (func = rule, tname = name, type = Real, scope = Real))
        iscomplex && push!(specs, (func = saferule, tname = name, type = ComplexF64, scope = ComplexF64))
        iscomplex && push!(specs, (func = rule, tname = name, type = Complex, scope = Complex))
        isnumber && push!(specs, (func = rule, tname = name, type = Number, scope = Number))
    end

    # datetime64
    push!(
        specs,
        (func = pyconvert_rule_datetime64, tname = "numpy:datetime64", type = DateTime64, scope = Any),
    )
    push!(specs, (func = pyconvert_rule_datetime64, tname = "numpy:datetime64", type = InlineDateTime64, scope = InlineDateTime64))
    push!(
        specs,
        (
            func = pyconvert_rule_datetime64,
            tname = "numpy:datetime64",
            type = NumpyDates.DatesInstant,
            scope = NumpyDates.DatesInstant,
        ),
    )
    push!(specs, (func = pyconvert_rule_datetime64, tname = "numpy:datetime64", type = Missing, scope = Missing))
    push!(specs, (func = pyconvert_rule_datetime64, tname = "numpy:datetime64", type = Nothing, scope = Nothing))

    # timedelta64
    push!(
        specs,
        (func = pyconvert_rule_timedelta64, tname = "numpy:timedelta64", type = TimeDelta64, scope = Any),
    )
    push!(specs, (func = pyconvert_rule_timedelta64, tname = "numpy:timedelta64", type = InlineTimeDelta64, scope = InlineTimeDelta64))
    push!(
        specs,
        (
            func = pyconvert_rule_timedelta64,
            tname = "numpy:timedelta64",
            type = NumpyDates.DatesPeriod,
            scope = NumpyDates.DatesPeriod,
        ),
    )
    push!(specs, (func = pyconvert_rule_timedelta64, tname = "numpy:timedelta64", type = Missing, scope = Missing))
    push!(specs, (func = pyconvert_rule_timedelta64, tname = "numpy:timedelta64", type = Nothing, scope = Nothing))

    return specs
end

function register_numpy_rules!()
    for rule in numpy_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
end
