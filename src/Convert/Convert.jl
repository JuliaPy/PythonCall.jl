"""
    module PythonCall.Convert

Implements `pyconvert`.
"""
module Convert

using ..PythonCall
using ..Core:
    Core,
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

import PythonCall:
    pyconvert, @pyconvert, pyconvert_add_rule, pyconvert_return, pyconvert_unconverted

# internal API
export pyconvert_add_rule,
    pyconvert_return,
    pyconvert_isunconverted,
    pyconvert_result,
    pyconvert_tryconvert,
    PYCONVERT_PRIORITY_ARRAY,
    PYCONVERT_PRIORITY_CANONICAL,
    PYCONVERT_PRIORITY_NORMAL

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
