"""
    module PythonCall.Convert

Implements `pyconvert`.
"""
module Convert

using ..PythonCall
using ..Utils
using ..C
using ..Core

using Dates: Date, Time, DateTime, Second, Millisecond, Microsecond, Nanosecond

import ..PythonCall:
    @pyconvert,
    pyconvert_add_rule,
    pyconvert_return,
    pyconvert_unconverted,
    pyconvert,
    PyConvertPriority

export
    pyconvert_isunconverted,
    pyconvert_result,
    pyconvert_result,
    pyconvert_tryconvert,
    pyconvert_unconverted,
    pyconvertarg


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
