const pydatetimetype = pylazyobject(() -> pydatetimemodule.datetime)
export pydatetimetype

pyisdatetime(o::PyObject) = pytypecheck(o, pydatetimetype)
export pyisdatetime

pydatetime(args...; opts...) = pydatetimetype(args...; opts...)
pydatetime(o::DateTime) = pydatetime(year(o), month(o), day(o), hour(o), minute(o), second(o), millisecond(o)*1000)
export pydatetime

const pydatetype = pylazyobject(() -> pydatetimemodule.date)
export pydatetype

pyisdate(o::PyObject) = pytypecheck(o, pydatetype)
export pyisdate

pydate(args...; opts...) = pydatetype(args...; opts...)
pydate(o::Date) = pydate(year(o), month(o), day(o))
export pydate

const pytimetype = pylazyobject(() -> pydatetimemodule.time)
export pytimetype

pyistime(o::PyObject) = pytypecheck(o, pytimetype)
export pyistime

pytime(args...; opts...) = pytimetype(args...; opts...)
pytime(o::Time) =
    if iszero(nanosecond(o))
        pytime(hour(o), minute(o), second(o), millisecond(o)*1000 + microsecond(o))
    else
        throw(InexactError(:pytime, PyObject, o))
    end
