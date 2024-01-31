module JLD2Ext

using JLD2
using PythonCall

### Py

struct Py_Serialized
    x::Union{Nothing,Vector{UInt8}}
end

JLD2.writeas(::Type{Py}) = Py_Serialized

function JLD2.wconvert(::Type{Py_Serialized}, x::Py)
    if PythonCall.pyisnull(x)
        v = nothing
    else
        v = copy(PythonCall._Py.pybytes_asvector(pyimport("pickle").dumps(x)))
    end
    Py_Serialized(v)
end

function JLD2.rconvert(::Type{Py}, x::Py_Serialized)
    v = x.x
    if v === nothing
        PythonCall.pynew()
    else
        pyimport("pickle").loads(pybytes(v))
    end
end

### PyException - handled similarly as with Serialization.jl

JLD2.writeas(::Type{PyException}) = Py_Serialized

JLD2.wconvert(::Type{Py_Serialized}, x::PyException) = JLD2.wconvert(Py_Serialized, x.v)

JLD2.rconvert(::Type{PyException}, x::Py_Serialized) = PyException(JLD2.rconvert(Py, x))

end
