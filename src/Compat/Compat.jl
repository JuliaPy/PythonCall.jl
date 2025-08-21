"""
    module PythonCall.Compat

Misc bits and bobs for compatibility.
"""
module Compat

using ..PythonCall
using ..Utils
using ..C
using ..Core
import ..Core: Py
using ..Wrap

using Serialization: Serialization, AbstractSerializer, serialize, deserialize
using Tables: Tables
using Requires: @require

import ..PythonCall:
    event_loop_on,
    event_loop_off,
    fix_qt_plugin_path,
    pytable

include("gui.jl")
include("ipython.jl")
include("multimedia.jl")
include("serialization.jl")
include("tables.jl")
include("pycall.jl")

function __init__()
    init_gui()
    init_pyshow()
    init_tables()
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)
end

end
