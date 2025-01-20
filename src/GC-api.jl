"""
    module PythonCall.GC

Garbage collection of Python objects.

See [`gc`](@ref).
"""
module GC

# public functions
for name in [:enable, :disable, :gc]
    @eval function $name end
    if Base.VERSION ≥ v"1.11"
        eval(Expr(:public, name))
    end
end

end
