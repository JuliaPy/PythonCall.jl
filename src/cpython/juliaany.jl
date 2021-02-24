const PyJuliaAnyValue_Type__ref = Ref(PyNULL)
PyJuliaAnyValue_Type() = begin
    ptr = PyJuliaAnyValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaBaseValue_Type()
        isnull(base) && return PyNULL
        t = fill(
            PyType_Create(
                c,
                name = "julia.AnyValue",
                base = base,
                repr = pyjlany_repr,
                str = pyjlany_str,
                getattro = pyjlany_getattro,
                setattro = pyjlany_setattro,
                call = pyjlany_call,
                iter = pyjlany_iter,
                richcompare = pyjlany_richcompare,
                as_mapping = (
                    length = pyjlany_length,
                    subscript = pyjlany_getitem,
                    ass_subscript = pyjlany_setitem,
                ),
                as_sequence = (contains = pyjlany_contains,),
                as_number = (
                    positive = pyjlany_positive,
                    negative = pyjlany_negative,
                    absolute = pyjlany_absolute,
                    power = pyjlany_power,
                    add = pyjlany_binop(+),
                    subtract = pyjlany_binop(-),
                    multiply = pyjlany_binop(*),
                    truedivide = pyjlany_binop(/),
                    divmod = pyjlany_binop((x, y) -> (fld(x, y), mod(x, y))),
                    floordivide = pyjlany_binop(fld),
                    remainder = pyjlany_binop(mod),
                    lshift = pyjlany_binop(<<),
                    rshift = pyjlany_binop(>>),
                    and = pyjlany_binop(&),
                    xor = pyjlany_binop(xor),
                    or = pyjlany_binop(|),
                ),
                methods = [
                    (name = "__dir__", flags = Py_METH_NOARGS, meth = pyjlany_dir),
                    (name = "_repr_mimebundle_", flags = Py_METH_VARARGS | Py_METH_KEYWORDS, meth = pyjlany_repr_mimebundle),
                    (name = "_jl_raw", flags = Py_METH_NOARGS, meth = pyjlany_toraw),
                    (name = "_jl_display", flags = Py_METH_NOARGS, meth = pyjlany_display),
                    (name = "_jl_help", flags = Py_METH_NOARGS, meth = pyjlany_help),
                ],
                getset = [(name = "__name__", get = pyjlany_name)],
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyNULL
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaAnyValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaAnyValue_New(x) = PyJuliaValue_New(PyJuliaAnyValue_Type(), x)
PyJuliaValue_From(x) = PyJuliaAnyValue_New(x)

pyjlany_repr(xo::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)
        s = repr(x)
        s = string("jl:", '\n' in s ? '\n' : ' ', s)
        PyUnicode_From(s)
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlany_str(xo::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)
        s = string(x)
        PyUnicode_From(s)
    catch err
        PyErr_SetJuliaError(err)
        return PyNULL
    end

pyjlany_getattro(xo::PyPtr, ko::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericGetAttr(xo, ko)
    if isnull(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    # Now try to get the corresponding property
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return PyNULL
    k = pyjl_attr_py2jl(k)
    try
        v = getproperty(x, Symbol(k))
        PyObject_From(v)
    catch err
        if !hasproperty(x, Symbol(k)) ||
           (err isa UndefVarError && err.var === Symbol(k)) ||
           (err isa ErrorException && occursin("has no field", err.msg))
            PyErr_SetStringFromJuliaError(PyExc_AttributeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end
end

propertytype(x, k) =
    propertiesarefields(typeof(x)) && hasfield(typeof(x), k) ? fieldtype(typeof(x), k) : Any
@generated propertiesarefields(::Type{T}) where {T} =
    which(getproperty, Tuple{T,Symbol}) == which(getproperty, Tuple{Any,Symbol})

pyjlany_setattro(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    # Try generic lookup first
    ro = PyObject_GenericSetAttr(xo, ko, vo)
    if ism1(ro) && PyErr_IsSet(PyExc_AttributeError())
        PyErr_Clear()
    else
        return ro
    end
    if isnull(vo)
        PyErr_SetString(PyExc_TypeError(), "attribute deletion not supported")
        return Cint(-1)
    end
    # Now try to set the corresponding property
    x = PyJuliaValue_GetValue(xo)
    k = PyUnicode_AsString(ko)
    isempty(k) && PyErr_IsSet() && return Cint(-1)
    k = pyjl_attr_py2jl(k)
    try
        V = propertytype(x, Symbol(k))
        ism1(PyObject_Convert(vo, V)) && return Cint(-1)
        v = takeresult(V)
        setproperty!(x, Symbol(k), v)
        Cint(0)
    catch err
        if !hasproperty(x, Symbol(k)) ||
           (err isa UndefVarError && err.var === Symbol(k)) ||
           (err isa ErrorException && occursin("has no field", err.msg))
            PyErr_SetStringFromJuliaError(PyExc_AttributeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Cint(-1)
    end
end

pyjl_dir(x) = propertynames(x)
pyjl_dir(x::Module) = begin
    r = Symbol[]
    append!(r, names(x, all = true, imported = true))
    for m in ccall(:jl_module_usings, Any, (Any,), x)::Vector
        append!(r, names(m))
    end
    r
end

pyjlany_dir(xo::PyPtr, _::PyPtr) = begin
    fo = PyObject_GetAttrString(PyJuliaBaseValue_Type(), "__dir__")
    isnull(fo) && return PyNULL
    ro = PyObject_CallNice(fo, PyObjectRef(xo))
    Py_DecRef(fo)
    isnull(ro) && return PyNULL
    x = PyJuliaValue_GetValue(xo)
    ks = try
        collect(map(string, pyjl_dir(x)))
    catch err
        Py_DecRef(ro)
        PyErr_SetJuliaError(err)
        return PyNULL
    end
    for k in ks
        ko = PyUnicode_From(pyjl_attr_jl2py(k))
        isnull(ko) && (Py_DecRef(ro); return PyNULL)
        err = PyList_Append(ro, ko)
        Py_DecRef(ko)
        ism1(err) && (Py_DecRef(ro); return PyNULL)
    end
    return ro
end

pyjlany_call(fo::PyPtr, argso::PyPtr, kwargso::PyPtr) = begin
    f = PyJuliaValue_GetValue(fo)
    if isnull(argso)
        args = Vector{Any}()
    else
        ism1(PyObject_Convert(argso, Vector{Any})) && return PyNULL
        args = takeresult(Vector{Any})
    end
    if isnull(kwargso)
        kwargs = Dict{Symbol,Any}()
    else
        ism1(PyObject_Convert(kwargso, Dict{Symbol,Any})) && return PyNULL
        kwargs = takeresult(Dict{Symbol,Any})
    end
    try
        x = f(args...; kwargs...)
        PyObject_From(x)
    catch err
        if err isa MethodError && err.f === f
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end
end

pyjlany_length(xo::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)
        Py_ssize_t(length(x))
    catch err
        if err isa MethodError && err.f === length
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Py_ssize_t(-1)
    end

@generated pyjl_keytype(::Type{T}) where {T} =
    try
        keytype(T)
    catch
        nothing
    end;
pyjl_hasvarindices(::Type) = true
pyjl_hasvarindices(::Type{<:AbstractDict}) = false

pyjl_getindices(x, ko) =
    if (K = pyjl_keytype(typeof(x))) !== nothing
        ism1(PyObject_Convert(ko, K)) ? PYERR() : (takeresult(K),)
    elseif pyjl_hasvarindices(typeof(x)) && PyTuple_Check(ko)
        ism1(PyObject_TryConvert(ko, Tuple)) ? PYERR() : takeresult(Tuple)
    else
        ism1(PyObject_TryConvert(ko, Any)) ? PYERR() : takeresult(Any)
    end

pyjlany_getitem(xo::PyPtr, ko::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    k = pyjl_getindices(x, ko)
    k === PYERR() && return PyNULL
    try
        PyObject_From(x[k...])
    catch err
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        elseif err isa KeyError && (err.key === k || (err.key,) === k)
            PyErr_SetStringFromJuliaError(PyExc_KeyError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end
end

@generated pyjl_valtype(::Type{T}) where {T} =
    try
        valtype(T)
    catch
        try
            eltype(T)
        catch
            nothing
        end
    end;

pyjl_getvalue(x, vo) =
    if (V = pyjl_valtype(typeof(x))) !== nothing
        ism1(PyObject_Convert(vo, V)) ? PYERR() : takeresult(V)
    else
        ism1(PyObject_Convert(vo, Any)) ? PYERR() : takeresult(Any)
    end

pyjlany_setitem(xo::PyPtr, ko::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    k = pyjl_getindices(x, ko)
    k === PYERR() && return Cint(-1)
    try
        if isnull(vo)
            delete!(x, k...)
            Cint(0)
        else
            v = pyjl_getvalue(x, vo)
            v === PYERR() && return Cint(-1)
            x[k...] = v
            Cint(0)
        end
    catch err
        if err isa BoundsError && err.a === x
            PyErr_SetStringFromJuliaError(PyExc_IndexError(), err)
        elseif err isa KeyError && (err.key === k || (err.key,) === k)
            PyErr_SetStringFromJuliaError(PyExc_KeyError(), err)
        elseif err isa MethodError && err.f === delete!
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Cint(-1)
    end
end

pyjlany_iter(xo::PyPtr) = PyJuliaIteratorValue_New(Iterator(PyJuliaValue_GetValue(xo)))

pyjlany_contains(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return Cint(-1)
    r == 0 && return Cint(0)
    v = takeresult(eltype(x))
    try
        Cint(v in x)
    catch err
        if err isa MethodError && err.f === :in
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        Cint(-1)
    end
end

pyjlany_richcompare(xo::PyPtr, yo::PyPtr, op::Cint) = begin
    x = PyJuliaValue_GetValue(xo)
    r = PyObject_TryConvert(yo, Any)
    r == -1 && return PyNULL
    r == 0 && return PyNotImplemented_New()
    y = takeresult()
    try
        if op == Py_EQ
            PyObject_From(x == y)
        elseif op == Py_NE
            PyObject_From(x != y)
        elseif op == Py_LE
            PyObject_From(x <= y)
        elseif op == Py_LT
            PyObject_From(x < y)
        elseif op == Py_GE
            PyObject_From(x >= y)
        elseif op == Py_GT
            PyObject_From(x > y)
        else
            PyErr_SetString(PyExc_ValueError(), "bad op given to richcompare: $op")
            PyNULL
        end
    catch err
        if err isa MethodError && err.f in (==, !=, <=, <, >=, >)
            PyNotImplemented_New()
        else
            PyErr_SetJuliaError(err)
            PyNULL
        end
    end
end

const ALL_MIMES = [
    "text/plain",
    "text/html",
    "text/markdown",
    "text/json",
    "text/latex",
    "text/xml",
    "text/csv",
    "application/javascript",
    "application/pdf",
    "application/ogg",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "audio/aac",
    "audio/mpeg",
    "audio/ogg",
    "audio/opus",
    "audio/webm",
    "audio/wav",
    "audio/midi",
    "audio/x-midi",
    "video/mpeg",
    "video/ogg",
    "video/webm",
]

pyjlany_repr_mimebundle(xo::PyPtr, args::PyPtr, kwargs::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    ism1(PyArg_CheckNumArgsEq("_repr_mimebundle_", args, 0)) && return PyNULL
    ism1(PyArg_GetArg(Union{Set{String},Nothing}, "_repr_mimebundle_", kwargs, "include", nothing)) && return PyNULL
    inc = takeresult(Union{Set{String},Nothing})
    ism1(PyArg_GetArg(Union{Set{String},Nothing}, "_repr_mimebundle_", kwargs, "exclude", nothing)) && return PyNULL
    exc = takeresult(Union{Set{String},Nothing})
    # decide which mimes to include
    if inc === nothing
        # default set of mimes to try
        mimes = copy(ALL_MIMES)
        # looks for mimes on show methods for the type
        for meth in methods(show, Tuple{IO, MIME, typeof(x)}).ms
            mimetype = meth.sig.parameters[3]
            mimetype isa DataType || continue
            mime = string(mimetype.parameters[1])
            push!(mimes, mime)
        end
    else
        mimes = push!(inc, "text/plain")
    end
    exc === nothing || setdiff!(mimes, exc)
    # make the bundle
    bundle = PyDict_New()
    isnull(bundle) && return PyNULL
    for m in mimes
        try
            io = IOBuffer()
            show(io, MIME(m), x)
            v = take!(io)
            mo = PyUnicode_From(m)
            isnull(mo) && (Py_DecRef(bundle); return PyNULL)
            vo = istextmime(m) ? PyUnicode_From(v) : PyBytes_From(v)
            isnull(vo) && (Py_DecRef(mo); Py_DecRef(bundle); return PyNULL)
            err = PyDict_SetItem(bundle, mo, vo)
            Py_DecRef(mo)
            Py_DecRef(vo)
            ism1(err) && (Py_DecRef(bundle); return PyNULL)
        catch err
            # silently skip anything that didn't work
        end
    end
    bundle
end

pyjlany_name(xo::PyPtr, ::Ptr{Cvoid}) =
    try
        PyObject_From(string(nameof(PyJuliaValue_GetValue(xo))))
    catch err
        if err isa MethodError && err.f === nameof
            PyErr_SetString(PyExc_AttributeError(), "__name__")
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end

pyjlany_toraw(xo::PyPtr, ::PyPtr) = PyJuliaRawValue_New(PyJuliaValue_GetValue(xo))

struct ExtraNewline{T}
    value :: T
end
Base.show(io::IO, m::MIME, x::ExtraNewline) = show(io, m, x.value)
Base.show(io::IO, m::MIME"text/plain", x::ExtraNewline) = (show(io, m, x.value); println(io))
Base.showable(m, x::ExtraNewline) = showable(m, x.value)

pyjlany_display(xo::PyPtr, ::PyPtr) = try
    x = PyJuliaValue_GetValue(xo)
    display(ExtraNewline(x))
    PyNone_New()
catch err
    PyErr_SetJuliaError(err)
    PyNULL
end

pyjlany_help(xo::PyPtr, ::PyPtr) =
    try
        x = Docs.doc(PyJuliaValue_GetValue(xo))
        display(ExtraNewline(x))
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlany_positive(xo::PyPtr) =
    try
        PyObject_From(+(PyJuliaValue_GetValue(xo)))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlany_negative(xo::PyPtr) =
    try
        PyObject_From(-(PyJuliaValue_GetValue(xo)))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlany_absolute(xo::PyPtr) =
    try
        PyObject_From(abs(PyJuliaValue_GetValue(xo)))
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

struct pyjlany_binop{F}
    f::F
end
(f::pyjlany_binop)(xo::PyPtr, yo::PyPtr) = begin
    PyJuliaValue_Check(xo) || return PyNotImplemented_New()
    PyJuliaValue_Check(yo) || return PyNotImplemented_New()
    x = PyJuliaValue_GetValue(xo)
    y = PyJuliaValue_GetValue(yo)
    try
        PyObject_From(f.f(x, y))
    catch err
        if err isa MethodError && err.f === f.f
            PyNotImplemented_New()
        else
            PyErr_SetJuliaError(err)
            PyNULL
        end
    end
end

pyjlany_power(xo::PyPtr, yo::PyPtr, zo::PyPtr) = begin
    PyJuliaValue_Check(xo) || return PyNotImplemented_New()
    PyJuliaValue_Check(yo) || return PyNotImplemented_New()
    x = PyJuliaValue_GetValue(xo)
    y = PyJuliaValue_GetValue(yo)
    if PyNone_Check(zo)
        try
            PyObject_From(x^y)
        catch err
            if err isa MethodError && err.f === ^
                PyNotImplemented_New()
            else
                PyErr_SetJuliaError(err)
                PyNULL
            end
        end
    else
        PyJuliaValue_Check(zo) || return PyNotImplemented_New()
        z = PyJuliaValue_GetValue(zo)
        try
            PyObject_From(powermod(x, y, z))
        catch err
            if err isa MethodError && err.f === powermod
                PyNotImplemented_New()
            else
                PyErr_SetJuliaError(err)
                PyNULL
            end
        end
    end
end
