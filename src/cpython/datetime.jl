for n in [:DateTime, :Date, :Time, :TimeDelta, :TZInfo, :TimeZone]
    p = lowercase(string(n))
    t = Symbol(:Py, n, :_Type)
    r = Symbol(t, :__ref)
    c = Symbol(:Py, n, :_Check)
    @eval $r = Ref(PyPtr())
    @eval $t(doimport::Bool=true) = begin
        ptr = $r[]
        isnull(ptr) || return ptr
        m = Py_DateTimeModule(doimport)
        isnull(m) && return PyPtr()
        o = PyObject_GetAttrString(m, $p)
        Py_DecRef(m)
        isnull(o) && return PyPtr()
        $r[] = o
    end
    @eval $c(o) = begin
        t = $t(false)
        isnull(t) && return (PyErr_IsSet() ? Cint(-1) : Cint(0))
        PyObject_IsInstance(o, t)
    end
end

PyDateTime_FromParts(year::Integer=1, month::Integer=1, day::Integer=1, hour::Integer=0, minute::Integer=0, second::Integer=0, microsecond::Integer=0; tzinfo=nothing, fold::Integer=0) = begin
    t = PyDateTime_Type()
    isnull(t) && return PyPtr()
    PyObject_CallNice(t, year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold)
end

PyDateTime_From(x::DateTime) =
    PyDateTime_FromParts(year(x), month(x), day(x), hour(x), minute(x), second(x), millisecond(x)*1000)
PyDateTime_From(x::Date) =
    PyDateTime_FromParts(year(x), month(x), day(x))

PyDateTime_IsAware(o::PyPtr) = begin
    tzinfo = PyObject_GetAttrString(o, "tzinfo")
    isnull(tzinfo) && return Cint(-1)
    PyNone_Check(tzinfo) && (Py_DecRef(tzinfo); return Cint(0))
    utcoffset = PyObject_GetAttrString(tzinfo, "utcoffset")
    Py_DecRef(tzinfo)
    isnull(utcoffset) && return Cint(-1)
    off = PyObject_CallNice(utcoffset, PyObjectRef(o))
    Py_DecRef(utcoffset)
    isnull(off) && return Cint(-1)
    r = PyNone_Check(off)
    Py_DecRef(off)
    r ? Cint(0) : Cint(1)
end

PyDateTime_TryConvertRule_datetime(o::PyPtr, ::Type{S}) where {S<:DateTime} = begin
    # TODO: worry about fold?
    # can only convert non-aware times
    aware = PyDateTime_IsAware(o)
    aware == -1 && return -1
    aware !=  0 && return  0
    # year
    p = PyObject_GetAttrString(o, "year")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    year = takeresult(Int)
    # month
    p = PyObject_GetAttrString(o, "month")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    month = takeresult(Int)
    # day
    p = PyObject_GetAttrString(o, "day")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    day = takeresult(Int)
    # hour
    p = PyObject_GetAttrString(o, "hour")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    hour = takeresult(Int)
    # minute
    p = PyObject_GetAttrString(o, "minute")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    min = takeresult(Int)
    # second
    p = PyObject_GetAttrString(o, "second")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    sec = takeresult(Int)
    # microsecond
    p = PyObject_GetAttrString(o, "microsecond")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    micro = takeresult(Int)
    # millisecond resolution
    mod(micro, 1000) == 0 || return 0
    # success
    putresult(DateTime(year, month, day, hour, min, sec, div(micro,1000)))
end

PyDate_FromParts(year::Integer=1, month::Integer=1, day::Integer=1) = begin
    t = PyDate_Type()
    isnull(t) && return PyPtr()
    PyObject_CallNice(t, year, month, day)
end

PyDate_From(x::Date) =
    PyDate_FromParts(year(x), month(x), day(x))

PyDate_TryConvertRule_date(o::PyPtr, ::Type{S}) where {S<:Date} = begin
    # year
    p = PyObject_GetAttrString(o, "year")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    year = takeresult(Int)
    # month
    p = PyObject_GetAttrString(o, "month")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    month = takeresult(Int)
    # day
    p = PyObject_GetAttrString(o, "day")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    day = takeresult(Int)
    # success
    putresult(Date(year, month, day))
end

PyTime_FromParts(hour::Integer=0, minute::Integer=0, second::Integer=0, microsecond::Integer=0; tzinfo=nothing, fold::Integer=0) = begin
    t = PyTime_Type()
    isnull(t) && return PyPtr()
    PyObject_CallNice(t, hour, minute, second, microsecond, tzinfo, fold=fold)
end

PyTime_From(x::Time) =
    if iszero(nanosecond(x))
        PyTime_FromParts(hour(x), minute(x), second(x), millisecond(x)*1000 + microsecond(x))
    else
        PyErr_SetString(PyExc_ValueError(), "cannot create 'datetime.time' with resolution less than microseconds")
        PyPtr()
    end

PyTime_IsAware(o::PyPtr) = begin
    tzinfo = PyObject_GetAttrString(o, "tzinfo")
    isnull(tzinfo) && return Cint(-1)
    PyNone_Check(tzinfo) && (Py_DecRef(tzinfo); return Cint(0))
    utcoffset = PyObject_GetAttrString(tzinfo, "utcoffset")
    Py_DecRef(tzinfo)
    isnull(utcoffset) && return Cint(-1)
    off = PyObject_CallNice(utcoffset, PyObjectRef(Py_None()))
    Py_DecRef(utcoffset)
    isnull(off) && return Cint(-1)
    r = PyNone_Check(off)
    Py_DecRef(off)
    r ? Cint(0) : Cint(1)
end

PyTime_TryConvertRule_time(o::PyPtr, ::Type{S}) where {S<:Time} = begin
    # TODO: worry about fold?
    # can only convert non-aware times
    aware = PyTime_IsAware(o)
    aware == -1 && return -1
    aware !=  0 && return  0
    # hour
    p = PyObject_GetAttrString(o, "hour")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    hour = takeresult(Int)
    # minute
    p = PyObject_GetAttrString(o, "minute")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    min = takeresult(Int)
    # second
    p = PyObject_GetAttrString(o, "second")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    sec = takeresult(Int)
    # microsecond
    p = PyObject_GetAttrString(o, "microsecond")
    isnull(p) && return -1
    r = PyObject_TryConvert(p, Int)
    Py_DecRef(p)
    r < 1 && return r
    micro = takeresult(Int)
    # success
    putresult(Time(hour, min, sec, div(micro,1000), mod(micro,1000)))
end

PyTimeDelta_FromParts(; days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0) = begin
    t = PyTimeDelta_Type()
    isnull(t) && return PyPtr()
    PyObject_CallNice(t, days, seconds, microseconds, milliseconds, minutes, hours, weeks)
end

PyTimeDelta_From(x::Week) = PyTimeDelta_FromParts(weeks=Dates.value(x))
PyTimeDelta_From(x::Day) = PyTimeDelta_FromParts(days=Dates.value(x))
PyTimeDelta_From(x::Hour) = PyTimeDelta_FromParts(hours=Dates.value(x))
PyTimeDelta_From(x::Minute) = PyTimeDelta_FromParts(minutes=Dates.value(x))
PyTimeDelta_From(x::Second) = PyTimeDelta_FromParts(seconds=Dates.value(x))
PyTimeDelta_From(x::Millisecond) = PyTimeDelta_FromParts(milliseconds=Dates.value(x))
PyTimeDelta_From(x::Microsecond) = PyTimeDelta_FromParts(microseconds=Dates.value(x))
PyTimeDelta_From(x::Nanosecond) =
    if mod(Dates.value(x), 1000) == 0
        PyTimeDelta_FromParts(div(Dates.value(x), 1000))
    else
        PyErr_SetString(PyExc_ValueError(), "cannot create 'datetime.timedelta' with resolution less than microseconds")
        PyPtr()
    end
PyTimeDelta_From(x::Period) = begin
    PyErr_SetString(PyExc_ValueError(), "cannot create 'datetime.timedelta' from a Julia '$(typeof(x))'")
    PyPtr()
end
PyTimeDelta_From(x::Dates.CompoundPeriod) = begin
    r = PyTimeDelta_FromParts()
    isnull(r) && return PyPtr()
    for p in x.periods
        po = PyTimeDelta_From(p)
        isnull(po) && (Py_DecRef(r); return PyPtr())
        r2 = PyNumber_Add(r, po)
        Py_DecRef(po)
        Py_DecRef(r)
        isnull(r2) && return PyPtr()
        r = r2
    end
    r
end

PyTimeDelta_AsParts(o::PyPtr) = begin
    # days
    p = PyObject_GetAttrString(o, "days")
    isnull(p) && return (-1,-1,-1)
    r = PyObject_Convert(p, Int)
    Py_DecRef(p)
    r == -1 && return (-1,-1,-1)
    days = takeresult(Int)
    # seconds
    p = PyObject_GetAttrString(o, "seconds")
    isnull(p) && return (-1,-1,-1)
    r = PyObject_Convert(p, Int)
    Py_DecRef(p)
    r == -1 && return (-1,-1,-1)
    seconds = takeresult(Int)
    # microseconds
    p = PyObject_GetAttrString(o, "microseconds")
    isnull(p) && return (-1,-1,-1)
    r = PyObject_Convert(p, Int)
    Py_DecRef(p)
    r == -1 && return (-1,-1,-1)
    microseconds = takeresult(Int)
    return (days, seconds, microseconds)
end

PyTimeDelta_TryConvertRule_compoundperiod(o::PyPtr, ::Type{S}) where {S<:Dates.CompoundPeriod} = begin
    days, seconds, microseconds = PyTimeDelta_AsParts(o)
    microseconds == -1 && return -1
    putresult(S(Period[Day(days), Second(seconds), Microsecond(microseconds)]))
end

PyTimeDelta_TryConvertRule_period(o::PyPtr, ::Type{S}) where {S<:Period} = begin
    # TODO: check for overflow
    days, seconds, microseconds = PyTimeDelta_AsParts(o)
    microseconds == -1 && return -1
    if Day <: S && seconds == 0 && microseconds == 0
        putresult(Day(days))
    elseif Week <: S && seconds == 0 && microseconds == 0 && mod(days, 7) == 0
        putresult(Week(div(days, 7)))
    elseif Second <: S && microseconds == 0
        putresult(Second(seconds + 60*60*24*days))
    elseif Millisecond <: S && mod(microseconds, 1000) == 0
        putresult(Millisecond(div(microseconds, 1000) + 1000*seconds + 1000*60*60*24*days))
    elseif Microsecond <: S
        putresult(Microsecond(microseconds + 1000_000*seconds + 1000_000*60*60*24*days))
    elseif Nanosecond <: S
        putresult(Nanosecond(1000*microseconds + 1000_000_000*seconds + 1000_000_000*60*60*24*days))
    else
        0
    end
end
