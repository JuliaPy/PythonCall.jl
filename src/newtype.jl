### CACHE

cacheptr!(c, x::Ptr) = x
cacheptr!(c, x::String) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractString) = cacheptr!(c, String(x))
cacheptr!(c, x::Array) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractArray) = cacheptr!(c, Array(x))
cacheptr!(c, x::PyObject) = (push!(c, x); pyptr(x))

### PROTOCOLS

cpynewnumbermethods!(c, x::C.PyNumberMethods) = x
cpynewnumbermethods!(c; opts...) = C.PyNumberMethods(; opts...)
cpynewnumbermethods!(c, x::Dict) = cpynewnumbermethods!(c; x...)
cpynewnumbermethods!(c, x::NamedTuple) = cpynewnumbermethods!(c; x...)

cpynewmappingmethods!(c, x::C.PyMappingMethods) = x
cpynewmappingmethods!(c; opts...) = C.PyMappingMethods(; opts...)
cpynewmappingmethods!(c, x::Dict) = cpynewmappingmethods!(c; x...)
cpynewmappingmethods!(c, x::NamedTuple) = cpynewmappingmethods!(c; x...)

cpynewsequencemethods!(c, x::C.PySequenceMethods) = x
cpynewsequencemethods!(c; opts...) = C.PySequenceMethods(; opts...)
cpynewsequencemethods!(c, x::Dict) = cpynewsequencemethods!(c; x...)
cpynewsequencemethods!(c, x::NamedTuple) = cpynewsequencemethods!(c; x...)

cpynewbufferprocs!(c, x::C.PyBufferProcs) = x
cpynewbufferprocs!(c; opts...) = C.PyBufferProcs(; opts...)
cpynewbufferprocs!(c, x::Dict) = cpynewbufferprocs!(c; x...)
cpynewbufferprocs!(c, x::NamedTuple) = cpynewbufferprocs!(c; x...)

cpynewmethod!(c, x::C.PyMethodDef) = x
cpynewmethod!(c; name=C_NULL, meth=C_NULL, flags=0, doc=C_NULL) =
    C.PyMethodDef(
        name = name isa Ptr ? name : cacheptr!(c, string(name)),
        meth = meth,
        flags = flags,
        doc = doc isa Ptr ? doc : cacheptr!(c, string(doc)),
    )
cpynewmethod!(c, x::Dict) = cpynewmethod!(c; x...)
cpynewmethod!(c, x::NamedTuple) = cpynewmethod!(c; x...)

cpynewgetset!(c, x::C.PyGetSetDef) = x
cpynewgetset!(c; name=C_NULL, get=C_NULL, set=C_NULL, doc=C_NULL, closure=C_NULL) =
    C.PyGetSetDef(
        name = name isa Ptr ? name : cacheptr!(c, string(name)),
        get = get,
        set = set,
        doc = doc isa Ptr ? doc : cacheptr!(c, string(doc)),
        closure = closure,
    )
cpynewgetset!(c, x::Dict) = cpynewgetset!(c; x...)
cpynewgetset!(c, x::NamedTuple) = cpynewgetset!(c; x...)

### NEW TYPE

function cpynewtype!(c; type=C_NULL, name, base=C_NULL, new=C.@pyglobal(:PyType_GenericNew), as_number=C_NULL, as_mapping=C_NULL, as_sequence=C_NULL, as_buffer=C_NULL, methods=C_NULL, getset=C_NULL, opts...)
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
            cacheptr!(c, [[cpynewmethod!(c, m) for m in methods]; C.PyMethodDef()])
        end
    getset =
        if getset isa Ptr
            getset
        else
            cacheptr!(c, [[cpynewgetset!(c, m) for m in getset]; C.PyGetSetDef()])
        end
    C.PyTypeObject(; ob_base=C.PyVarObject(ob_base=C.PyObject(type=type)), name=name, base=base, new=new, as_number=as_number, as_mapping=as_mapping, as_sequence=as_sequence, as_buffer=as_buffer, methods, getset, opts...)
end
