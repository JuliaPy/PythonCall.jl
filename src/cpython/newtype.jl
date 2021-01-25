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
        let c = $(esc(c)), f = $(esc(f))
            if f isa Ptr
                f
            elseif f isa Base.CFunction
                cacheptr!(c, f)
            else
                cacheptr!(
                    c,
                    @cfunction($(Expr(:$, :f)), $(esc(R)), ($(map(esc, Ts.args)...),))
                )
            end
        end
    end
end

### PROTOCOLS

PyNumberMethods_Create(c, x::PyNumberMethods) = x
PyNumberMethods_Create(
    c;
    add = C_NULL,
    subtract = C_NULL,
    multiply = C_NULL,
    remainder = C_NULL,
    divmod = C_NULL,
    power = C_NULL,
    negative = C_NULL,
    positive = C_NULL,
    absolute = C_NULL,
    bool = C_NULL,
    invert = C_NULL,
    lshift = C_NULL,
    rshift = C_NULL,
    and = C_NULL,
    or = C_NULL,
    xor = C_NULL,
    int = C_NULL,
    float = C_NULL,
    inplace_add = C_NULL,
    inplace_subtract = C_NULL,
    inplace_multiply = C_NULL,
    inplace_remainder = C_NULL,
    inplace_power = C_NULL,
    inplace_lshift = C_NULL,
    inplace_rshift = C_NULL,
    inplace_and = C_NULL,
    inplace_xor = C_NULL,
    inplace_or = C_NULL,
    floordivide = C_NULL,
    truedivide = C_NULL,
    inplace_floordivide = C_NULL,
    inplace_truedivide = C_NULL,
    index = C_NULL,
    matrixmultiply = C_NULL,
    inplace_matrixmultiply = C_NULL,
) = PyNumberMethods(
    add = @cachefuncptr!(c, add, PyPtr, (PyPtr, PyPtr)),
    subtract = @cachefuncptr!(c, subtract, PyPtr, (PyPtr, PyPtr)),
    multiply = @cachefuncptr!(c, multiply, PyPtr, (PyPtr, PyPtr)),
    remainder = @cachefuncptr!(c, remainder, PyPtr, (PyPtr, PyPtr)),
    divmod = @cachefuncptr!(c, divmod, PyPtr, (PyPtr, PyPtr)),
    power = @cachefuncptr!(c, power, PyPtr, (PyPtr, PyPtr, PyPtr)),
    negative = @cachefuncptr!(c, negative, PyPtr, (PyPtr,)),
    positive = @cachefuncptr!(c, positive, PyPtr, (PyPtr,)),
    absolute = @cachefuncptr!(c, absolute, PyPtr, (PyPtr,)),
    bool = @cachefuncptr!(c, bool, Cint, (PyPtr,)),
    invert = @cachefuncptr!(c, invert, PyPtr, (PyPtr,)),
    lshift = @cachefuncptr!(c, lshift, PyPtr, (PyPtr, PyPtr)),
    rshift = @cachefuncptr!(c, rshift, PyPtr, (PyPtr, PyPtr)),
    and = @cachefuncptr!(c, and, PyPtr, (PyPtr, PyPtr)),
    xor = @cachefuncptr!(c, xor, PyPtr, (PyPtr, PyPtr)),
    or = @cachefuncptr!(c, or, PyPtr, (PyPtr, PyPtr)),
    int = @cachefuncptr!(c, int, PyPtr, (PyPtr,)),
    float = @cachefuncptr!(c, float, PyPtr, (PyPtr,)),
    inplace_add = @cachefuncptr!(c, inplace_add, PyPtr, (PyPtr, PyPtr)),
    inplace_subtract = @cachefuncptr!(c, inplace_subtract, PyPtr, (PyPtr, PyPtr)),
    inplace_multiply = @cachefuncptr!(c, inplace_multiply, PyPtr, (PyPtr, PyPtr)),
    inplace_remainder = @cachefuncptr!(c, inplace_remainder, PyPtr, (PyPtr, PyPtr)),
    inplace_power = @cachefuncptr!(c, inplace_power, PyPtr, (PyPtr, PyPtr, PyPtr)),
    inplace_lshift = @cachefuncptr!(c, inplace_lshift, PyPtr, (PyPtr, PyPtr)),
    inplace_rshift = @cachefuncptr!(c, inplace_rshift, PyPtr, (PyPtr, PyPtr)),
    inplace_and = @cachefuncptr!(c, inplace_and, PyPtr, (PyPtr, PyPtr)),
    inplace_xor = @cachefuncptr!(c, inplace_xor, PyPtr, (PyPtr, PyPtr)),
    inplace_or = @cachefuncptr!(c, inplace_or, PyPtr, (PyPtr, PyPtr)),
    floordivide = @cachefuncptr!(c, floordivide, PyPtr, (PyPtr, PyPtr)),
    truedivide = @cachefuncptr!(c, truedivide, PyPtr, (PyPtr, PyPtr)),
    inplace_floordivide = @cachefuncptr!(c, inplace_floordivide, PyPtr, (PyPtr, PyPtr)),
    inplace_truedivide = @cachefuncptr!(c, inplace_truedivide, PyPtr, (PyPtr, PyPtr)),
    index = @cachefuncptr!(c, index, PyPtr, (PyPtr,)),
    matrixmultiply = @cachefuncptr!(c, matrixmultiply, PyPtr, (PyPtr, PyPtr)),
    inplace_matrixmultiply = @cachefuncptr!(
        c,
        inplace_matrixmultiply,
        PyPtr,
        (PyPtr, PyPtr)
    ),
)
PyNumberMethods_Create(c, x::Dict) = PyNumberMethods_Create(c; x...)
PyNumberMethods_Create(c, x::NamedTuple) = PyNumberMethods_Create(c; x...)

PyMappingMethods_Create(c, x::PyMappingMethods) = x
PyMappingMethods_Create(c; length = C_NULL, subscript = C_NULL, ass_subscript = C_NULL) =
    PyMappingMethods(
        length = @cachefuncptr!(c, length, Py_ssize_t, (PyPtr,)),
        subscript = @cachefuncptr!(c, subscript, PyPtr, (PyPtr, PyPtr)),
        ass_subscript = @cachefuncptr!(c, ass_subscript, Cint, (PyPtr, PyPtr, PyPtr)),
    )
PyMappingMethods_Create(c, x::Dict) = PyMappingMethods_Create(c; x...)
PyMappingMethods_Create(c, x::NamedTuple) = PyMappingMethods_Create(c; x...)

PySequenceMethods_Create(c, x::PySequenceMethods) = x
PySequenceMethods_Create(
    c;
    length = C_NULL,
    concat = C_NULL,
    repeat = C_NULL,
    item = C_NULL,
    ass_item = C_NULL,
    contains = C_NULL,
    inplace_concat = C_NULL,
    inplace_repeat = C_NULL,
) = PySequenceMethods(
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
PyBufferProcs_Create(c; get = C_NULL, release = C_NULL) = PyBufferProcs(
    get = @cachefuncptr!(c, get, Cint, (PyPtr, Ptr{Py_buffer}, Cint)),
    release = @cachefuncptr!(c, release, Cvoid, (PyPtr, Ptr{Py_buffer})),
)
PyBufferProcs_Create(c, x::Dict) = PyBufferProcs_Create(c; x...)
PyBufferProcs_Create(c, x::NamedTuple) = PyBufferProcs_Create(c; x...)

PyMethodDef_Create(c, x::PyMethodDef) = x
PyMethodDef_Create(c; name = C_NULL, meth = C_NULL, flags = 0, doc = C_NULL) = PyMethodDef(
    name = cachestrptr!(c, name),
    meth = iszero(flags & Py_METH_KEYWORDS) ?
           @cachefuncptr!(c, meth, PyPtr, (PyPtr, PyPtr)) :
           @cachefuncptr!(c, meth, PyPtr, (PyPtr, PyPtr, PyPtr)),
    flags = flags,
    doc = cachestrptr!(c, doc),
)
PyMethodDef_Create(c, x::Dict) = PyMethodDef_Create(c; x...)
PyMethodDef_Create(c, x::NamedTuple) = PyMethodDef_Create(c; x...)

PyGetSetDef_Create(c, x::PyGetSetDef) = x
PyGetSetDef_Create(
    c;
    name = C_NULL,
    get = C_NULL,
    set = C_NULL,
    doc = C_NULL,
    closure = C_NULL,
) = PyGetSetDef(
    name = cachestrptr!(c, name),
    get = @cachefuncptr!(c, get, PyPtr, (PyPtr, Ptr{Cvoid})),
    set = @cachefuncptr!(c, set, Cint, (PyPtr, PyPtr, Ptr{Cvoid})),
    doc = cachestrptr!(c, doc),
    closure = closure,
)
PyGetSetDef_Create(c, x::Dict) = PyGetSetDef_Create(c; x...)
PyGetSetDef_Create(c, x::NamedTuple) = PyGetSetDef_Create(c; x...)

PyType_Create(
    c;
    type = C_NULL,
    name,
    as_number = C_NULL,
    as_mapping = C_NULL,
    as_sequence = C_NULL,
    as_buffer = C_NULL,
    methods = C_NULL,
    getset = C_NULL,
    dealloc = C_NULL,
    getattr = C_NULL,
    setattr = C_NULL,
    repr = C_NULL,
    hash = C_NULL,
    call = C_NULL,
    str = C_NULL,
    getattro = C_NULL,
    setattro = C_NULL,
    doc = C_NULL,
    iter = C_NULL,
    iternext = C_NULL,
    richcompare = C_NULL,
    opts...,
) = begin
    type = cacheptr!(c, type)
    name = cachestrptr!(c, name)
    as_number =
        as_number isa Ptr ? as_number :
        cacheptr!(c, fill(PyNumberMethods_Create(c, as_number)))
    as_mapping =
        as_mapping isa Ptr ? as_mapping :
        cacheptr!(c, fill(PyMappingMethods_Create(c, as_mapping)))
    as_sequence =
        as_sequence isa Ptr ? as_sequence :
        cacheptr!(c, fill(PySequenceMethods_Create(c, as_sequence)))
    as_buffer =
        as_buffer isa Ptr ? as_buffer :
        cacheptr!(c, fill(PyBufferProcs_Create(c, as_buffer)))
    methods = if methods isa Ptr
        methods
    else
        cacheptr!(c, [[PyMethodDef_Create(c, m) for m in methods]; PyMethodDef()])
    end
    getset = if getset isa Ptr
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
        ob_base = PyVarObject(ob_base = PyObject(type = type)),
        name = name,
        as_number = as_number,
        as_mapping = as_mapping,
        as_sequence = as_sequence,
        as_buffer = as_buffer,
        methods = methods,
        getset = getset,
        dealloc = dealloc,
        getattr = getattr,
        setattr = setattr,
        repr = repr,
        hash = hash,
        call = call,
        str = str,
        getattro = getattro,
        setattro = setattro,
        iter = iter,
        iternext = iternext,
        richcompare = richcompare,
        opts...,
    )
end
