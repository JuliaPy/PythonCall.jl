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
    priority = 1
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

        if isnumber
            pyconvert_add_rule(name, Number, Number, rule)
            if isreal
                pyconvert_add_rule(name, Real, Number, rule)
                pyconvert_add_rule(name, AbstractFloat, Number, rule)
            else
                pyconvert_add_rule(name, Complex, Number, rule)
            end
            if isint
                pyconvert_add_rule(name, Integer, Number, rule)
                pyconvert_add_rule(name, isuint ? Unsigned : Signed, Number, rule)
                if isuint
                    pyconvert_add_rule(
                        name,
                        UInt,
                        Number,
                        sizeof(T) ≤ sizeof(UInt) ? saferule : rule,
                    )
                else
                    pyconvert_add_rule(
                        name,
                        Int,
                        Number,
                        sizeof(T) ≤ sizeof(Int) ? saferule : rule,
                    )
                end
            elseif isfloat
                pyconvert_add_rule(name, Float64, Number, saferule)
            elseif iscomplex
                pyconvert_add_rule(name, ComplexF64, Number, saferule)
            end
            pyconvert_add_rule(name, T, Number, saferule)
        end
    end

    # datetime64
    pyconvert_add_rule("numpy:datetime64", InlineDateTime64, InlineDateTime64, pyconvert_rule_datetime64)
    pyconvert_add_rule("numpy:datetime64",
        NumpyDates.DatesInstant,
        NumpyDates.DatesInstant,
        pyconvert_rule_datetime64)
    pyconvert_add_rule("numpy:datetime64", Missing, Missing, pyconvert_rule_datetime64)
    pyconvert_add_rule("numpy:datetime64", Nothing, Nothing, pyconvert_rule_datetime64)

    # timedelta64
    pyconvert_add_rule("numpy:timedelta64", InlineTimeDelta64, InlineTimeDelta64, pyconvert_rule_timedelta64)
    pyconvert_add_rule("numpy:timedelta64",
        NumpyDates.DatesPeriod,
        NumpyDates.DatesPeriod,
        pyconvert_rule_timedelta64)
    pyconvert_add_rule("numpy:timedelta64", Missing, Missing, pyconvert_rule_timedelta64)
    pyconvert_add_rule("numpy:timedelta64", Nothing, Nothing, pyconvert_rule_timedelta64)
    for (t, T) in NUMPY_SIMPLE_TYPES
        pyconvert_add_rule_high_priority(
            "numpy:$t",
            T,
            Any,
            pyconvert_rule_numpysimplevalue{T,true}(),
            priority,
        )
    end
    pyconvert_add_rule_high_priority(
        "numpy:datetime64",
        DateTime64,
        Any,
        pyconvert_rule_datetime64,
        priority,
    )
    pyconvert_add_rule_high_priority(
        "numpy:timedelta64",
        TimeDelta64,
        Any,
        pyconvert_rule_timedelta64,
        priority,
    )
end
