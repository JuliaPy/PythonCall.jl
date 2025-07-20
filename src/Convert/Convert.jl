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
    pyconvert,
    @pyconvert,
    pyconvert_add_rule,
    pyconvert_return,
    pyconvert_unconverted,
    PyConvertPriority,
    PYCONVERT_PRIORITY_WRAP,
    PYCONVERT_PRIORITY_ARRAY,
    PYCONVERT_PRIORITY_CANONICAL,
    PYCONVERT_PRIORITY_NORMAL,
    PYCONVERT_PRIORITY_FALLBACK

export
    pyconvert_tryconvert,
    pyconvertarg,
    pyconvert_result,
    pyconvert_unconverted,
    pyconvert_isunconverted,
    pyconvert_result


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
