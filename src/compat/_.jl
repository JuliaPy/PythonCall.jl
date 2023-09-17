"""
    module _compat

Misc bits and bobs for compatibility.
"""
module _compat
    using .._Py
    using .._Py: C, pynew
    using Tables: Tables

    # include("compat/gui.jl")
    # include("compat/ipython.jl")
    # include("compat/multimedia.jl")
    # include("compat/serialization.jl")
    # include("compat/tables.jl")

    function __init__()
        C.with_gil() do
            # init_gui()
            # init_pyshow()
            # init_tables()
        end
    end
end
