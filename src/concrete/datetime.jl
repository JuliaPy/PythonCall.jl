# We used to use 1/1/1 but pandas.Timestamp is a subclass of datetime and does not include
# this date, so we use 1970 instead.
const _base_datetime = DateTime(1970, 1, 1)
const _base_pydatetime = pynew()

function init_datetime()
    pycopy!(_base_pydatetime, pydatetimetype(1970, 1, 1))
end

pydate(year, month, day) = pydatetype(year, month, day)
pydate(x::Date) = pydate(year(x), month(x), day(x))
export pydate

pytime(_hour=0, _minute=0, _second=0, _microsecond=0, _tzinfo=nothing; hour=_hour, minute=_minute, second=_second, microsecond=_microsecond, tzinfo=_tzinfo, fold=0) = pytimetype(hour, minute, second, microsecond, tzinfo, fold=fold)
pytime(x::Time) =
    if iszero(nanosecond(x))
        pytime(hour(x), minute(x), second(x), millisecond(x) * 1000 + microsecond(x))
    else
        errset(pybuiltins.ValueError, "cannot create 'datetime.time' with less than microsecond resolution")
        pythrow()
    end
export pytime

pydatetime(year, month, day, _hour=0, _minute=0, _second=0, _microsecond=0, _tzinfo=nothing; hour=_hour, minute=_minute, second=_second, microsecond=_microsecond, tzinfo=_tzinfo, fold=0) = pydatetimetype(year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold)
function pydatetime(x::DateTime)
    # compute time since _base_datetime
    # this accounts for fold
    d = pytimedeltatype(milliseconds = (x - _base_datetime).value)
    ans = _base_pydatetime + d
    pydel!(d)
    return ans
end
pydatetime(x::Date) = pydatetime(year(x), month(x), day(x))
export pydatetime

function pytime_isaware(x)
    tzinfo = pygetattr(x, "tzinfo")
    if pyisnone(tzinfo)
        pydel!(tzinfo)
        return false
    end
    utcoffset = tzinfo.utcoffset
    pydel!(tzinfo)
    o = utcoffset(nothing)
    pydel!(utcoffset)
    ans = !pyisnone(o)
    pydel!(o)
    return ans
end

function pydatetime_isaware(x)
    tzinfo = pygetattr(x, "tzinfo")
    if pyisnone(tzinfo)
        pydel!(tzinfo)
        return false
    end
    utcoffset = tzinfo.utcoffset
    pydel!(tzinfo)
    o = utcoffset(x)
    pydel!(utcoffset)
    ans = !pyisnone(o)
    pydel!(o)
    return ans
end

function pyconvert_rule_date(::Type{Date}, x::Py)
    # datetime is a subtype of date, but we shouldn't convert datetime to Date since it's lossy
    pyisinstance(x, pydatetimetype) && return pyconvert_unconverted()
    year = pyconvert_and_del(Int, x.year)
    month = pyconvert_and_del(Int, x.month)
    day = pyconvert_and_del(Int, x.day)
    pyconvert_return(Date(year, month, day))
end

function pyconvert_rule_time(::Type{Time}, x::Py)
    pytime_isaware(x) && return pyconvert_unconverted()
    hour = pyconvert_and_del(Int, x.hour)
    minute = pyconvert_and_del(Int, x.minute)
    second = pyconvert_and_del(Int, x.second)
    microsecond = pyconvert_and_del(Int, x.microsecond)
    return pyconvert_return(Time(hour, minute, second, div(microsecond, 1000), mod(microsecond, 1000)))
end

function pyconvert_rule_datetime(::Type{DateTime}, x::Py)
    pydatetime_isaware(x) && return pyconvert_unconverted()
    # compute the time since _base_datetime
    # this accounts for fold
    d = x - _base_pydatetime
    days = pyconvert_and_del(Int, d.days)
    seconds = pyconvert_and_del(Int, d.seconds)
    microseconds = pyconvert_and_del(Int, d.microseconds)
    pydel!(d)
    iszero(mod(microseconds, 1000)) || return pyconvert_unconverted()
    return pyconvert_return(_base_datetime + Millisecond(div(microseconds, 1000) + 1000 * (seconds + 60 * 60 * 24 * days)))
end
