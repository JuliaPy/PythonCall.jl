
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
