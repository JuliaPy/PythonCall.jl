### CACHE

cacheptr!(c, x::Ptr) = x
cacheptr!(c, x::String) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractString) = cacheptr!(c, String(x))
cacheptr!(c, x::Array) = (push!(c, x); pointer(x))
cacheptr!(c, x::AbstractArray) = cacheptr!(c, Array(x))
cacheptr!(c, x::PyObject) = (push!(c, x); pyptr(x))
cacheptr!(c, x::Base.CFunction) = (push!(c, x); Base.unsafe_convert(Ptr{Cvoid}, x))

cachestrptr!(c, x::Ptr) = x
cachestrptr!(c, x) = cacheptr!(c, string(x))

macro cachefuncptr!(c, f, R, Ts)
    quote
        let c=$(esc(c)), f=$(esc(f))
            if f isa Ptr
                f
            elseif f isa Base.CFunction
                cacheptr!(c, f)
            else
                cacheptr!(c, @cfunction($(Expr(:$, :f)), $(esc(R)), ($(map(esc, Ts.args)...),)))
            end
        end
    end
end

### PROTOCOLS

PyNumberMethods_Create(c, x::PyNumberMethods) = x
PyNumberMethods_Create(c; opts...) = C.PyNumberMethods(; [k => (v isa Ptr ? v : v isa Base.CFunction ? cacheptr!(c, v) : error()) for (k,v) in pairs(opts)]...)
PyNumberMethods_Create(c, x::Dict) = PyNumberMethods_Create(c; x...)
PyNumberMethods_Create(c, x::NamedTuple) = PyNumberMethods_Create(c; x...)

PyMappingMethods_Create(c, x::PyMappingMethods) = x
PyMappingMethods_Create(c; length=C_NULL, subscript=C_NULL, ass_subscript=C_NULL) =
    PyMappingMethods(
        length = @cachefuncptr!(c, length, Py_ssize_t, (PyPtr,)),
        subscript = @cachefuncptr!(c, subscript, PyPtr, (PyPtr, PyPtr)),
        ass_subscript = @cachefuncptr!(c, ass_subscript, Cint, (PyPtr, PyPtr, PyPtr)),
    )
PyMappingMethods_Create(c, x::Dict) = PyMappingMethods_Create(c; x...)
PyMappingMethods_Create(c, x::NamedTuple) = PyMappingMethods_Create(c; x...)

PySequenceMethods_Create(c, x::PySequenceMethods) = x
PySequenceMethods_Create(c;
    length=C_NULL, concat=C_NULL, repeat=C_NULL, item=C_NULL, ass_item=C_NULL,
    contains=C_NULL, inplace_concat=C_NULL, inplace_repeat=C_NULL,
) =
    PySequenceMethods(
        length = @cachefuncptr!(c, length, Py_ssize_t, (PyPtr,)),
        concat = @cachefuncptr!(c, concat, PyPtr, (PyPtr, PyPtr)),
        repeat = @cachefuncptr!(c, repeat, PyPtr, (PyPtr, Py_ssize_t)),
        item = @cachefuncptr!(c, item, PyPtr, (PyPtr, Py_ssize_t)),
        ass_item = @cachefuncptr!(c, ass_item, Cint, (PyPtr, Py_ssize_t, PyPtr)),
        contains = @cachefuncptr!(c, contains, Cint, (PyPtr, PyPtr)),
        inplace_concat = @cachefuncptr!(c, inplace_concat, PyPtr, (PyPtr, PyPtr)),
        inplace_repeat = @cachefuncptr!(c, inplace_repeat, PyPtr, (PyPtr, Py_ssize_t)),
    )
PySequenceMethods_Create(c, x::Dict) = PySequenceMethods_Create(c; x...)
PySequenceMethods_Create(c, x::NamedTuple) = PySequenceMethods_Create(c; x...)

PyBufferProcs_Create(c, x::PyBufferProcs) = x
PyBufferProcs_Create(c; get=C_NULL, release=C_NULL) =
    PyBufferProcs(
        get = @cachefuncptr!(c, get, Cint, (PyPtr, Ptr{Py_buffer}, Cint)),
        release = @cachefuncptr!(c, release, Cvoid, (PyPtr, Ptr{Py_buffer})),
    )
PyBufferProcs_Create(c, x::Dict) = PyBufferProcs_Create(c; x...)
PyBufferProcs_Create(c, x::NamedTuple) = PyBufferProcs_Create(c; x...)

PyMethodDef_Create(c, x::PyMethodDef) = x
PyMethodDef_Create(c; name=C_NULL, meth=C_NULL, flags=0, doc=C_NULL) =
    PyMethodDef(
        name = cachestrptr!(c, name),
        meth = iszero(flags & Py_METH_KEYWORDS) ? @cachefuncptr!(c, meth, PyPtr, (PyPtr, PyPtr)) : @cachefuncptr!(c, meth, PyPtr, (PyPtr, PyPtr, PyPtr)),
        flags = flags,
        doc = cachestrptr!(c, doc),
    )
PyMethodDef_Create(c, x::Dict) = PyMethodDef_Create(c; x...)
PyMethodDef_Create(c, x::NamedTuple) = PyMethodDef_Create(c; x...)

PyGetSetDef_Create(c, x::PyGetSetDef) = x
PyGetSetDef_Create(c; name=C_NULL, get=C_NULL, set=C_NULL, doc=C_NULL, closure=C_NULL) =
    PyGetSetDef(
        name = cachestrptr!(c, name),
        get = @cachefuncptr!(c, get, PyPtr, (PyPtr, Ptr{Cvoid})),
        set = @cachefuncptr!(c, set, Cint, (PyPtr, PyPtr, Ptr{Cvoid})),
        doc = cachestrptr!(c, doc),
        closure = closure,
    )
PyGetSetDef_Create(c, x::Dict) = PyGetSetDef_Create(c; x...)
PyGetSetDef_Create(c, x::NamedTuple) = PyGetSetDef_Create(c; x...)

PyType_Create(c;
    type=C_NULL, name, as_number=C_NULL, as_mapping=C_NULL, as_sequence=C_NULL,
    as_buffer=C_NULL, methods=C_NULL, getset=C_NULL, dealloc=C_NULL, getattr=C_NULL,
    setattr=C_NULL, repr=C_NULL, hash=C_NULL, call=C_NULL, str=C_NULL, getattro=C_NULL,
    setattro=C_NULL, doc=C_NULL, iter=C_NULL, iternext=C_NULL, richcompare=C_NULL, opts...
) = begin
    type = cacheptr!(c, type)
    name = cachestrptr!(c, name)
    as_number = as_number isa Ptr ? as_number : cacheptr!(c, fill(PyNumberMethods_Create(c, as_number)))
    as_mapping = as_mapping isa Ptr ? as_mapping : cacheptr!(c, fill(PyMappingMethods_Create(c, as_mapping)))
    as_sequence = as_sequence isa Ptr ? as_sequence : cacheptr!(c, fill(PySequenceMethods_Create(c, as_sequence)))
    as_buffer =  as_buffer isa Ptr ? as_buffer : cacheptr!(c, fill(PyBufferProcs_Create(c, as_buffer)))
    methods =
        if methods isa Ptr
            methods
        else
            cacheptr!(c, [[PyMethodDef_Create(c, m) for m in methods]; PyMethodDef()])
        end
    getset =
        if getset isa Ptr
            getset
        else
            cacheptr!(c, [[PyGetSetDef_Create(c, m) for m in getset]; PyGetSetDef()])
        end
    dealloc = @cachefuncptr!(c, dealloc, Cvoid, (PyPtr,))
    getattr = @cachefuncptr!(c, getattr, PyPtr, (PyPtr, Cstring))
    setattr = @cachefuncptr!(c, setattr, Cint, (PyPtr, Cstring, PyPtr))
    repr = @cachefuncptr!(c, repr, PyPtr, (PyPtr,))
    hash = @cachefuncptr!(c, hash, Py_hash_t, (PyPtr,))
    call = @cachefuncptr!(c, call, PyPtr, (PyPtr, PyPtr, PyPtr))
    str = @cachefuncptr!(c, str, PyPtr, (PyPtr,))
    getattro = @cachefuncptr!(c, getattro, PyPtr, (PyPtr, PyPtr))
    setattro = @cachefuncptr!(c, setattro, Cint, (PyPtr, PyPtr, PyPtr))
    doc = cachestrptr!(c, doc)
    iter = @cachefuncptr!(c, iter, PyPtr, (PyPtr,))
    iternext = @cachefuncptr!(c, iternext, PyPtr, (PyPtr,))
    richcompare = @cachefuncptr!(c, richcompare, PyPtr, (PyPtr, PyPtr, Cint))
    PyTypeObject(;
        ob_base=PyVarObject(ob_base=PyObject(type=type)), name=name, as_number=as_number,
        as_mapping=as_mapping, as_sequence=as_sequence, as_buffer=as_buffer,
        methods=methods, getset=getset, dealloc=dealloc, getattr=getattr, setattr=setattr,
        repr=repr, hash=hash, call=call, str=str, getattro=getattro, setattro=setattro,
        iter=iter, iternext=iternext, richcompare=richcompare, opts...
    )
end
