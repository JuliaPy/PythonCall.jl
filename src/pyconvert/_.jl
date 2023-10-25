"""
    module _pyconvert

Implements `pyconvert`.
"""
module _pyconvert

using .._Py
using .._Py: C, Utils, @autopy, getptr, incref, pynew, PyNULL, pyisnull, pydel!, pyisint, iserrset_ambig, pyisnone, pyisTrue, pyisFalse, pyfloat_asdouble, pycomplex_ascomplex, pyisstr, pystr_asstring, pyisbytes, pybytes_asvector, pybytes_asUTF8string, pyisfloat, pyisrange, pytuple_getitem, unsafe_pynext, pyistuple, pydatetimetype, pytime_isaware, pydatetime_isaware, _base_pydatetime, _base_datetime, errmatches, errclear, errset, pyiscomplex, pythrow, pybool_asbool
using Dates: Date, Time, DateTime, Millisecond

import .._Py: pyconvert

include("pyconvert.jl")
include("rules.jl")
include("ctypes.jl")
include("numpy.jl")
include("pandas.jl")

function __init__()
    C.with_gil() do 
        init_pyconvert()
        init_ctypes()
        init_numpy()
        init_pandas()
    end
end

end
