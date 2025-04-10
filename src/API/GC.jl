"""
    module PythonCall.GC

Garbage collection of Python objects.

See [`gc`](@ref).
"""
module GC

# functions
function enable end
function disable end
function gc end

# public bindings
if Base.VERSION ≥ v"1.11"
    eval(Expr(:public, :enable, :disable, :gc))
end

end
