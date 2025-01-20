"The version of PythonCall."
const VERSION = v"0.9.23"

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

# public functions
for name in
    [:python_executable_path, :python_library_handle, :python_library_path, :python_version]
    @eval function $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end

# other public bindings
for name in [:VERSION, :GIL]
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end
