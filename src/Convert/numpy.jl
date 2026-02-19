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

