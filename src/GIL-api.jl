"""
    module PythonCall.GIL

Handling the Python Global Interpreter Lock.

See [`lock`](@ref), [`@lock`](@ref), [`unlock`](@ref) and [`@unlock`](@ref).
"""
module GIL

# public functions
for name in [:lock, :unlock]
    @eval function $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end

# public macros
for name in [:lock, :unlock]
    @eval macro $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, Symbol("@", name)))
    end
end

end
