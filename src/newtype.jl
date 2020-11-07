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

cpynewmethod!(c, x::CPyMethodDefStruct) = x
cpynewmethod!(c; name=C_NULL, meth=C_NULL, flags=0, doc=C_NULL) =
    CPyMethodDefStruct(
        name = name isa Ptr ? name : cacheptr!(c, string(name)),
        meth = meth,
        flags = flags,
        doc = doc isa Ptr ? doc : cacheptr!(c, string(doc)),
    )
cpynewmethod!(c, x::Dict) = cpynewmethod!(c; x...)
cpynewmethod!(c, x::NamedTuple) = cpynewmethod!(c; x...)

cpynewgetset!(c, x::CPyGetSetDefStruct) = x
cpynewgetset!(c; name=C_NULL, get=C_NULL, set=C_NULL, doc=C_NULL, closure=C_NULL) =
    CPyGetSetDefStruct(
        name = name isa Ptr ? name : cacheptr!(c, string(name)),
        get = get,
        set = set,
        doc = doc isa Ptr ? doc : cacheptr!(c, string(doc)),
        closure = closure,
    )
cpynewgetset!(c, x::Dict) = cpynewgetset!(c; x...)
cpynewgetset!(c, x::NamedTuple) = cpynewgetset!(c; x...)

### NEW TYPE

function cpynewtype!(c; type=C_NULL, name, base=C_NULL, new=cglobal((:PyType_GenericNew, PYLIB)), as_number=C_NULL, as_mapping=C_NULL, as_sequence=C_NULL, as_buffer=C_NULL, methods=C_NULL, getset=C_NULL, opts...)
    type = cacheptr!(c, type)
    name = cacheptr!(c, name)
    base = cacheptr!(c, base)
    as_number = as_number isa Ptr ? as_number : cacheptr!(c, fill(cpynewnumbermethods!(c, as_number)))
    as_mapping = as_mapping isa Ptr ? as_mapping : cacheptr!(c, fill(cpynewmappingmethods!(c, as_mapping)))
    as_sequence = as_sequence isa Ptr ? as_sequence : cacheptr!(c, fill(cpynewsequencemethods!(c, as_sequence)))
    as_buffer =  as_buffer isa Ptr ? as_buffer : cacheptr!(c, fill(cpynewbufferprocs!(c, as_buffer)))
    methods =
        if methods isa Ptr
            methods
        else
            cacheptr!(c, [[cpynewmethod!(c, m) for m in methods]; CPyMethodDefStruct()])
        end
    getset =
        if getset isa Ptr
            getset
        else
            cacheptr!(c, [[cpynewgetset!(c, m) for m in getset]; CPyGetSetDefStruct()])
        end
    CPyTypeObject(; ob_base=CPyVarObject(ob_base=CPyObject(type=type)), name=name, base=base, new=new, as_number=as_number, as_mapping=as_mapping, as_sequence=as_sequence, as_buffer=as_buffer, methods, getset, opts...)
end
