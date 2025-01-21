"""
    module PythonCall.Convert

Implements `pyconvert`.
"""
module Convert

using ..Core
using ..Core:
    C,
    Utils,
    @autopy,
    getptr,
    incref,
    pynew,
    PyNULL,
    pyisnull,
    pydel!,
    pyisint,
    iserrset_ambig,
    pyisnone,
    pyisTrue,
    pyisFalse,
    pyfloat_asdouble,
    pycomplex_ascomplex,
    pyisstr,
    pystr_asstring,
    pyisbytes,
    pybytes_asvector,
    pybytes_asUTF8string,
    pyisfloat,
    pyisrange,
    pytuple_getitem,
    unsafe_pynext,
    pyistuple,
    pydatetimetype,
    pytime_isaware,
    pydatetime_isaware,
    _base_pydatetime,
    _base_datetime,
    errmatches,
    errclear,
    errset,
    pyiscomplex,
    pythrow,
    pybool_asbool
using Dates: Date, Time, DateTime, Second, Millisecond, Microsecond, Nanosecond
using Dates: Year, Month, Day, Hour, Minute, Week, Period, CompoundPeriod, canonicalize

import ..Core: pyconvert

# patch conversion to Period types for julia <= 1.7
@static if VERSION < v"1.8.0-"
    for T in (:Year, :Month, :Week, :Day, :Hour, :Minute, :Second, :Millisecond, :Microsecond, :Nanosecond)
        @eval Base.convert(::Type{$T}, x::CompoundPeriod) = sum($T, x.periods; init = zero($T))
    end
end

include("pyconvert.jl")
include("rules.jl")
include("ctypes.jl")
include("numpy.jl")
include("pandas.jl")

function __init__()
    init_pyconvert()
    init_ctypes()
    init_numpy()
    init_pandas()
end

end
