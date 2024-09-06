struct pyconvert_rule_numpysimplevalue{R,S} <: Function end

function (::pyconvert_rule_numpysimplevalue{R,SAFE})(::Type{T}, x::Py) where {R,SAFE,T}
    ans = Base.GC.@preserve x C.PySimpleObject_GetValue(R, getptr(x))
    if SAFE
        pyconvert_return(convert(T, ans))
    else
        pyconvert_tryconvert(T, ans)
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

function pydatetime64(
    _year::Integer=0, _month::Integer=1, _day::Integer=1, _hour::Integer=0, _minute::Integer=0,_second::Integer=0, _millisecond::Integer=0, _microsecond::Integer=0, _nanosecond::Integer=0;
    year::Integer=_year, month::Integer=_month, day::Integer=_day, hour::Integer=_hour, minute::Integer=_minute, second::Integer=_second,
    millisecond::Integer=_millisecond, microsecond::Integer=_microsecond, nanosecond::Integer=_nanosecond
)
    pyimport("numpy").datetime64("$(DateTime(year, month, day, hour, minute, second))") + pytimedelta64(;millisecond, microsecond, nanosecond)
end
function pydatetime64(@nospecialize(x::T)) where T <: Period
    T <: Union{Week, Day, Hour, Minute, Second, Millisecond, Microsecond} || 
        error("Unsupported Period type: `$x::$T`. Consider using pytimedelta64 instead.")
    args = map(Base.Fix1(isa, x), (Day, Second, Millisecond, Microsecond,  Minute, Hour, Week))
    pydatetime64(map(Base.Fix1(*, x.value), args)...)
end
function pydatetime64(x::CompoundPeriod)
    x =  canonicalize(x)
    isempty(x.periods) ? pydatetime64(Second(0)) : sum(pydatetime64, x.periods)
end
export pydatetime64

function pytimedelta64(
    _year::Integer=0, _month::Integer=0, _day::Integer=0, _hour::Integer=0, _minute::Integer=0, _second::Integer=0, _millisecond::Integer=0, _microsecond::Integer=0, _nanosecond::Integer=0, _week::Integer=0;
    year::Integer=_year, month::Integer=_month, day::Integer=_day, hour::Integer=_hour, minute::Integer=_minute, second::Integer=_second, microsecond::Integer=_microsecond, millisecond::Integer=_millisecond, nanosecond::Integer=_nanosecond, week::Integer=_week)
    pytimedelta64(sum((
        Year(year), Month(month),
        # you cannot mix year or month with any of the below units in python
        # in case of wrong usage a descriptive error message will by thrown by the underlying python function
        Day(day), Hour(hour), Minute(minute), Second(second), Millisecond(millisecond), Microsecond(microsecond), Nanosecond(nanosecond), Week(week))
    ))
end
function pytimedelta64(@nospecialize(x::T)) where T <: Period
    index = findfirst(==(T), (Year, Month, Week, Day, Hour, Minute, Second, Millisecond, Microsecond, Nanosecond, T))::Int
    unit = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "")[index]
    pyimport("numpy").timedelta64(x.value, unit)
end
function pytimedelta64(x::CompoundPeriod)
    x =  canonicalize(x)
    isempty(x.periods) ? pytimedelta64(Second(0)) : sum(pytimedelta64.(x.periods))
end
export pytimedelta64

function pyconvert_rule_datetime64(::Type{DateTime}, x::Py)
    unit, count = pyconvert(Tuple, pyimport("numpy").datetime_data(x))
    value = reinterpret(Int64, pyconvert(Vector, x))[1]
    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")
    types = (Year, Month, Week, Day, Hour, Minute, Second, Millisecond, Microsecond, Nanosecond)
    T = types[findfirst(==(unit), units)::Int]
    pyconvert_return(DateTime(_base_datetime) + T(value * count))
end

function pyconvert_rule_timedelta64(::Type{CompoundPeriod}, x::Py)
    unit, count = pyconvert(Tuple, pyimport("numpy").datetime_data(x))
    value = reinterpret(Int64, pyconvert(Vector, x))[1]
    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")
    types = (Year, Month, Week, Day, Hour, Minute, Second, Millisecond, Microsecond, Nanosecond)
    T = types[findfirst(==(unit), units)::Int]
    pyconvert_return(CompoundPeriod(T(value * count)) |> canonicalize)
end

function pyconvert_rule_timedelta64(::Type{T}, x::Py) where T<:Period
    pyconvert_return(convert(T, pyconvert_rule_timedelta64(CompoundPeriod, x)))
end

function init_numpy()
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

        pyconvert_add_rule(name, T, saferule, PYCONVERT_PRIORITY_ARRAY)
        isuint && pyconvert_add_rule(name, UInt, sizeof(T) ≤ sizeof(UInt) ? saferule : rule)
        isuint && pyconvert_add_rule(name, Int, sizeof(T) < sizeof(Int) ? saferule : rule)
        isint &&
            !isuint &&
            pyconvert_add_rule(name, Int, sizeof(T) ≤ sizeof(Int) ? saferule : rule)
        isint && pyconvert_add_rule(name, Integer, rule)
        isfloat && pyconvert_add_rule(name, Float64, saferule)
        isreal && pyconvert_add_rule(name, Real, rule)
        iscomplex && pyconvert_add_rule(name, ComplexF64, saferule)
        iscomplex && pyconvert_add_rule(name, Complex, rule)
        isnumber && pyconvert_add_rule(name, Number, rule)
    end

    priority = PYCONVERT_PRIORITY_ARRAY
    pyconvert_add_rule("numpy:datetime64", DateTime, pyconvert_rule_datetime64, priority)
    let TT = (CompoundPeriod, Year, Month, Day, Hour, Minute, Second, Millisecond, Microsecond, Nanosecond, Week)
        Base.Cartesian.@nexprs 11 i -> pyconvert_add_rule("numpy:timedelta64", TT[i], pyconvert_rule_timedelta64, priority)
    end

    priority = PYCONVERT_PRIORITY_CANONICAL
    pyconvert_add_rule("numpy:datetime64", DateTime, pyconvert_rule_datetime64, priority)
    pyconvert_add_rule("numpy:timedelta64", Nanosecond, pyconvert_rule_timedelta, priority)    
end
