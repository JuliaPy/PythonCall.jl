"""
    module _compat

Misc bits and bobs for compatibility.
"""
module _compat
    using .._Py
    using .._Py: C, Utils, pynew, incref, getptr, pycopy!, pymodulehooks, pyisnull, pybytes_asvector, pysysmodule, pyosmodule, pystr_fromUTF8
    using .._pyconvert: pyconvert, @pyconvert
    using .._pywrap: PyPandasDataFrame
    using Serialization: Serialization, AbstractSerializer, serialize, deserialize
    using Tables: Tables
    using Requires: @require

    include("gui.jl")
    include("ipython.jl")
    include("multimedia.jl")
    include("serialization.jl")
    include("tables.jl")
    include("pycall.jl")

    function __init__()
        C.with_gil() do
            init_gui()
            init_pyshow()
            init_tables()
        end
        @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)
    end
end
