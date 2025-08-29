"""
    module PythonCall.Compat

Misc bits and bobs for compatibility.
"""
module Compat

using ..PythonCall
using ..Utils
using ..C
using ..Core
using ..Wrap

using Serialization: Serialization, AbstractSerializer, serialize, deserialize
using Tables: Tables

import ..PythonCall: event_loop_on, event_loop_off, fix_qt_plugin_path, pytable

include("gui.jl")
include("ipython.jl")
include("multimedia.jl")
include("serialization.jl")
include("tables.jl")

function __init__()
    init_gui()
    init_pyshow()
end

end
