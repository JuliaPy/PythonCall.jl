"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).
"""
module GIL

# functions
function lock end
function unlock end

# macros
macro lock end
macro unlock end

# public bindings
if Base.VERSION ≥ v"1.11"
    eval(Expr(:public, :lock, :unlock, Symbol("@lock"), Symbol("@unlock")))
end

end
