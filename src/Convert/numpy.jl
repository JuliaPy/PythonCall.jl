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

function init_numpy()
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

        pyconvert_add_rule(saferule, name, T, Any)
        isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(UInt) ? saferule : rule, name, UInt)
        isuint && pyconvert_add_rule(sizeof(T) < sizeof(Int) ? saferule : rule, name, Int)
        isint &&
            !isuint &&
            pyconvert_add_rule(sizeof(T) ≤ sizeof(Int) ? saferule : rule, name, Int)
        isint && pyconvert_add_rule(rule, name, Integer)
        isfloat && pyconvert_add_rule(saferule, name, Float64)
        isreal && pyconvert_add_rule(rule, name, Real)
        iscomplex && pyconvert_add_rule(saferule, name, ComplexF64)
        iscomplex && pyconvert_add_rule(rule, name, Complex)
        isnumber && pyconvert_add_rule(rule, name, Number)
    end

    # datetime64
    pyconvert_add_rule(
        pyconvert_rule_datetime64,
        "numpy:datetime64",
        DateTime64,
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_datetime64, "numpy:datetime64", InlineDateTime64)
    pyconvert_add_rule(
        pyconvert_rule_datetime64,
        "numpy:datetime64",
        NumpyDates.DatesInstant,
    )
    pyconvert_add_rule(pyconvert_rule_datetime64, "numpy:datetime64", Missing, Missing)
    pyconvert_add_rule(pyconvert_rule_datetime64, "numpy:datetime64", Nothing, Nothing)

    # timedelta64
    pyconvert_add_rule(
        pyconvert_rule_timedelta64,
        "numpy:timedelta64",
        TimeDelta64,
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_timedelta64, "numpy:timedelta64", InlineTimeDelta64)
    pyconvert_add_rule(
        pyconvert_rule_timedelta64,
        "numpy:timedelta64",
        NumpyDates.DatesPeriod,
    )
    pyconvert_add_rule(pyconvert_rule_timedelta64, "numpy:timedelta64", Missing, Missing)
    pyconvert_add_rule(pyconvert_rule_timedelta64, "numpy:timedelta64", Nothing, Nothing)
end
