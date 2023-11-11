"""
    module PythonCall.JlCore

Implements the Python type `juliacall.Jl` for wrapping Julia values. Exports [`pyjl`](@ref).
"""
module JlCore

using ..Core
using ..Core: Core, errcheck
using ..Convert: pyconvert_add_rule, pyconvert_tryconvert, PYCONVERT_PRIORITY_WRAP

function pyjl end

include("C.jl")

"""
    pyjl(v)

Wrap the Julia value `v` as a Python `juliacall.Jl`.
"""
pyjl(@nospecialize(v)) = pynew(errcheck(JlC.PyJl_New(v)))
export pyjl

"""
    pyjlvalue(x)

Extract the Julia value from the given Python `juliacall.Jl`.
"""
function pyjlvalue(x)
    x = Py(x)
    JlC.PyJl_Check(x) || throw(PyException(pybuiltins.TypeError("expecting a 'juliacall.Jl' but got a '$(pytype(x))'")))
    JlC.PyJl_GetValue(x)
end
export pyjlvalue

"""
    pyisjl(x)

Test whether the given Python object is a `juliacall.Jl`.
"""
function pyisjl(x)
    x = Py(x)
    JlC.PyJl_Check(x)
end
export pyisjl

# the fallback conversion
Core.Py(x) = pyjl(x)

const pyjltype = pynew()

pyconvert_rule_jl(::Type{T}, x::Py) where {T} = pyconvert_tryconvert(T, pyjlvalue(x))

function __init__()
    JlC.C.with_gil() do 
        Core.unsafe_setptr!(pyjltype, JlC.PyJl_Type[])
        jl = Core.pyjuliacallmodule
        jl.Jl = pyjltype
        jl.Core = pyjl(Base.Core)
        jl.Base = pyjl(Base)
        jl.Main = pyjl(Main)
    end
    pyconvert_add_rule("juliacall:Jl", Any, pyconvert_rule_jl, PYCONVERT_PRIORITY_WRAP)
end

end
