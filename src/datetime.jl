const pydatetimemodule = PyLazyObject(() -> pyimport("datetime"))

const pydatetimetype = PyLazyObject(() -> pydatetimemodule.datetime)
export pydatetimetype

pyisdatetime(o::AbstractPyObject) = pytypecheck(o, pydatetimetype)
export pyisdatetime

pydatetime(args...; opts...) = pydatetimetype(args...; opts...)
pydatetime(o::DateTime) = pydatetime(year(o), month(o), day(o), hour(o), minute(o), second(o), millisecond(o)*1000)
export pydatetime

const pydatetype = PyLazyObject(() -> pydatetimemodule.date)
export pydatetype

pyisdate(o::AbstractPyObject) = pytypecheck(o, pydatetype)
export pyisdate

pydate(args...; opts...) = pydatetype(args...; opts...)
pydate(o::Date) = pydate(year(o), month(o), day(o))
export pydate

const pytimetype = PyLazyObject(() -> pydatetimemodule.time)
export pytimetype

pyistime(o::AbstractPyObject) = pytypecheck(o, pytimetype)
export pyistime

pytime(args...; opts...) = pytimetype(args...; opts...)
pytime(o::Time) =
    if iszero(nanosecond(o))
        pytime(hour(o), minute(o), second(o), millisecond(o)*1000 + microsecond(o))
    else
        throw(InexactError(:pytime, PyObject, o))
    end
