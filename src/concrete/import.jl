"""
    pyimport(m)
    pyimport(m => k)
    pyimport(m => (k1, k2, ...))
    pyimport(m1, m2, ...)

Import a module `m`, or an attribute `k`, or a tuple of attributes.

If several arguments are given, return the results of importing each one in a tuple.
"""
pyimport(m) = pynew(errcheck(@autopy m C.PyImport_Import(getptr(m_))))
pyimport((m,k)::Pair) = (m_=pyimport(m); k_=pygetattr(m_,k); pydel!(m_); k_)
pyimport((m,ks)::Pair{<:Any,<:Tuple}) = (m_=pyimport(m); ks_=map(k->pygetattr(m_,k), ks); pydel!(m_); ks_)
pyimport(m1, m2, ms...) = map(pyimport, (m1, m2, ms...))
export pyimport

"""
    pyimport_conda(m, package; [channel])

Equivalent to `pyimport(m)` except that on failure we install the given `package` using conda and try again.
"""
function pyimport_conda(m, package; channel="")
    if Conda.available()
        try
            pyimport(m)
        catch
            Conda.add(package, channel=channel)
            pyimport(m)
        end
    else
        pyimport(m)
    end
end
export pyimport_conda
