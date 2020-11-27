"""
    pyimport(m, ...; [condapkg], [condachannel])
    pyimport(m => k, ...)

Import and return the module `m`.

If additionally `k` is given, then instead returns this attribute from `m`. If it is a tuple, a tuple of attributes is returned.

If two or more arguments are given, they are all imported and returned as a tuple.

If there is no such module and `condapkg` is given, then the given package will be automatically installed. Additionally, `condachannel` specifies the channel.
"""
function pyimport(m; condapkg::Union{Nothing,AbstractString}=nothing, condachannel::AbstractString="")
    if condapkg === nothing || !CONFIG.isconda
        m isa AbstractString ? check(C.PyImport_ImportModule(m)) : check(C.PyImport_Import(pyobject(m)))
    else
        try
            pyimport(m)
        catch err
            if err isa PythonRuntimeError && pyissubclass(err.t, pymodulenotfounderror)
                Conda.add(condapkg, CONFIG.condaenv; channel=condachannel)
                pyimport(m)
            else
                rethrow()
            end
        end
    end
end
function pyimport(x::Pair; opts...)
    m = pyimport(x[1]; opts...)
    x[2] isa Tuple ? map(k->pygetattr(m, k), x[2]) : pygetattr(m, x[2])
end
pyimport(m1, m2, ms...) = (pyimport(m1), pyimport(m2), map(pyimport, ms)...)
export pyimport
