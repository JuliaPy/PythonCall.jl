### CACHE

cacheptr!(c, x::Ptr) = x
cacheptr!(c, x::String) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractString) = cacheptr!(c, String(x))
cacheptr!(c, x::Array) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractArray) = cacheptr!(c, Array(x))
cacheptr!(c, x::AbstractPyObject) = (push!(c, x); pyptr(x))

### PROTOCOLS

cpynewnumbermethods!(c, x::CPyNumberMethodsStruct) = x
cpynewnumbermethods!(c; opts...) = CPyNumberMethodsStruct(; opts...)
cpynewnumbermethods!(c, x::Dict) = cpynewnumbermethods!(c; x...)
cpynewnumbermethods!(c, x::NamedTuple) = cpynewnumbermethods!(c; x...)

cpynewmappingmethods!(c, x::CPyMappingMethodsStruct) = x
cpynewmappingmethods!(c; opts...) = CPyMappingMethodsStruct(; opts...)
cpynewmappingmethods!(c, x::Dict) = cpynewmappingmethods!(c; x...)
cpynewmappingmethods!(c, x::NamedTuple) = cpynewmappingmethods!(c; x...)

cpynewsequencemethods!(c, x::CPySequenceMethodsStruct) = x
cpynewsequencemethods!(c; opts...) = CPySequenceMethodsStruct(; opts...)
cpynewsequencemethods!(c, x::Dict) = cpynewsequencemethods!(c; x...)
cpynewsequencemethods!(c, x::NamedTuple) = cpynewsequencemethods!(c; x...)

cpynewbufferprocs!(c, x::CPyBufferProcs) = x
cpynewbufferprocs!(c; opts...) = CPyBufferProcs(; opts...)
cpynewbufferprocs!(c, x::Dict) = cpynewbufferprocs!(c; x...)
cpynewbufferprocs!(c, x::NamedTuple) = cpynewbufferprocs!(c; x...)

### NEW TYPE

function cpynewtype!(c; type=C_NULL, name, base=C_NULL, new=cglobal((:PyType_GenericNew, PYLIB)), as_number=C_NULL, as_mapping=C_NULL, as_sequence=C_NULL, as_buffer=C_NULL, opts...)
    type = cacheptr!(c, type)
    name = cacheptr!(c, name)
    base = cacheptr!(c, base)
    as_number = as_number isa Ptr ? as_number : cacheptr!(c, fill(cpynewnumbermethods!(c, as_number)))
    as_mapping = as_mapping isa Ptr ? as_mapping : cacheptr!(c, fill(cpynewmappingmethods!(c, as_mapping)))
    as_sequence = as_sequence isa Ptr ? as_sequence : cacheptr!(c, fill(cpynewsequencemethods!(c, as_sequence)))
    as_buffer =  as_buffer isa Ptr ? as_buffer : cacheptr!(c, fill(cpynewbufferprocs!(c, as_buffer)))
    CPyTypeObject(; ob_base=CPyVarObject(ob_base=CPyObject(type=type)), name=name, base=base, new=new, as_number=as_number, as_mapping=as_mapping, as_sequence=as_sequence, as_buffer=as_buffer, opts...)
end
