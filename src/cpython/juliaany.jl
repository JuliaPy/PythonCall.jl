pyjlany_repr(xo::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)
        s = "<jl $(repr(x))>"
        PyUnicode_From(s)
    end PyNULL

pyjlany_str(xo::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)
        s = string(x)
        PyUnicode_From(s)
    end PyNULL

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
    @pyjltry begin
        v = getproperty(x, Symbol(k))
        PyObject_From(v)
    end PyNULL (Custom, !hasproperty(x, Symbol(k)))=>AttributeError (UndefVarError, Symbol(k))=>AttributeError (ErrorException, r"has no field")=>AttributeError
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
    @pyjltry begin
        V = propertytype(x, Symbol(k))
        ism1(PyObject_Convert(vo, V)) && return Cint(-1)
        v = takeresult(V)
        setproperty!(x, Symbol(k), v)
        Cint(0)
    end Cint(-1) (Custom, !hasproperty(x, Symbol(k)))=>AttributeError (UndefVarError, Symbol(k))=>AttributeError (ErrorException, r"has no field")=>AttributeError
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
    ks = @pyjltry collect(map(string, pyjl_dir(x))) PyNULL OnErr=>Py_DecRef(ro)
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
    @pyjltry PyObject_From(f(args...; kwargs...)) PyNULL (MethodError, f)=>TypeError
end

pyjlany_length(xo::PyPtr) =
    @pyjltry Py_ssize_t(length(PyJuliaValue_GetValue(xo))) Py_ssize_t(-1) (MethodError, length)=>TypeError

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
    @pyjltry PyObject_From(x[k...]) PyNULL BoundsError=>IndexError KeyError=>KeyError
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
    @pyjltry begin
        if isnull(vo)
            delete!(x, k...)
            Cint(0)
        else
            v = pyjl_getvalue(x, vo)
            v === PYERR() && return Cint(-1)
            x[k...] = v
            Cint(0)
        end
    end Cint(-1) BoundsError=>IndexError KeyError=>KeyError (MethodError, delete!)=>TypeError
end

pyjlany_iter(xo::PyPtr) = PyJuliaIteratorValue_New(Iterator(PyJuliaValue_GetValue(xo)))

pyjlany_contains(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return Cint(-1)
    r == 0 && return Cint(0)
    v = takeresult(eltype(x))
    @pyjltry Cint(v in x) Cint(-1) (MethodError, in)=>TypeError
end

pyjlany_richcompare(xo::PyPtr, yo::PyPtr, op::Cint) = begin
    x = PyJuliaValue_GetValue(xo)
    r = PyObject_TryConvert(yo, Any)
    r == -1 && return PyNULL
    r == 0 && return PyNotImplemented_New()
    y = takeresult()
    @pyjltry begin
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
    end PyNULL (MethodError, ==, !=, <=, <, >=, >)=>NotImplemented
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

mimes_for(@nospecialize(x)) = begin
    # default mimes we always try
    mimes = copy(ALL_MIMES)
    # look for mimes on show methods for this type
    for meth in methods(show, Tuple{IO, MIME, typeof(x)}).ms
        mimetype = meth.sig.parameters[3]
        mimetype isa DataType || continue
        mime = string(mimetype.parameters[1])
        push!(mimes, mime)
    end
    return mimes
end

pyjlany_repr_mimebundle(xo::PyPtr, args::PyPtr, kwargs::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)
    ism1(PyArg_CheckNumArgsEq("_repr_mimebundle_", args, 0)) && return PyNULL
    ism1(PyArg_GetArg(Union{Set{String},Nothing}, "_repr_mimebundle_", kwargs, "include", nothing)) && return PyNULL
    inc = takeresult(Union{Set{String},Nothing})
    ism1(PyArg_GetArg(Union{Set{String},Nothing}, "_repr_mimebundle_", kwargs, "exclude", nothing)) && return PyNULL
    exc = takeresult(Union{Set{String},Nothing})
    # decide which mimes to include
    mimes = inc === nothing ? mimes_for(x) : push!(inc, "text/plain")
    # respect exclude
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
    @pyjltry PyObject_From(string(nameof(PyJuliaValue_GetValue(xo)))) PyNULL (MethodError, nameof)=>AttributeError

pyjlany_toraw(xo::PyPtr, ::PyPtr) = PyJuliaRawValue_New(PyJuliaValue_GetValue(xo))

struct ExtraNewline{T}
    value :: T
end
Base.show(io::IO, m::MIME, x::ExtraNewline) = show(io, m, x.value)
Base.show(io::IO, m::MIME"text/plain", x::ExtraNewline) = (show(io, m, x.value); println(io))
Base.showable(m, x::ExtraNewline) = showable(m, x.value)

pyjlany_display(xo::PyPtr, ::PyPtr) =
    @pyjltry begin
        x = PyJuliaValue_GetValue(xo)
        display(ExtraNewline(x))
        PyNone_New()
    end PyNULL

pyjlany_help(xo::PyPtr, ::PyPtr) =
    @pyjltry begin
        x = Docs.doc(PyJuliaValue_GetValue(xo))
        display(ExtraNewline(x))
        PyNone_New()
    end PyNULL

pyjlany_positive(xo::PyPtr) =
    @pyjltry PyObject_From(+(PyJuliaValue_GetValue(xo))) PyNULL

pyjlany_negative(xo::PyPtr) =
    @pyjltry PyObject_From(-(PyJuliaValue_GetValue(xo))) PyNULL

pyjlany_absolute(xo::PyPtr) =
    @pyjltry PyObject_From(abs(PyJuliaValue_GetValue(xo))) PyNULL

struct pyjlany_binop{F}
    f::F
end
(f::pyjlany_binop)(xo::PyPtr, yo::PyPtr) = begin
    PyJuliaValue_Check(xo) || return PyNotImplemented_New()
    PyJuliaValue_Check(yo) || return PyNotImplemented_New()
    x = PyJuliaValue_GetValue(xo)
    y = PyJuliaValue_GetValue(yo)
    @pyjltry PyObject_From(f.f(x, y)) PyNULL (MethodError, f.f)=>NotImplemented
end

pyjlany_power(xo::PyPtr, yo::PyPtr, zo::PyPtr) = begin
    PyJuliaValue_Check(xo) || return PyNotImplemented_New()
    PyJuliaValue_Check(yo) || return PyNotImplemented_New()
    x = PyJuliaValue_GetValue(xo)
    y = PyJuliaValue_GetValue(yo)
    if PyNone_Check(zo)
        @pyjltry PyObject_From(x^y) PyNULL (MethodError, ^)=>NotImplemented
    else
        PyJuliaValue_Check(zo) || return PyNotImplemented_New()
        z = PyJuliaValue_GetValue(zo)
        @pyjltry PyObject_From(powermod(x, y, z)) PyNULL (MethodError, powermod)=>NotImplemented
    end
end

PyJuliaAnyValue_New(x) = PyJuliaValue_New(PyJuliaAnyValue_Type(), x)
PyJuliaValue_From(x) = PyJuliaAnyValue_New(x)

const PyJuliaAnyValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaBaseValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.AnyValue"),
        base = base,
        repr = @cfunctionOO(pyjlany_repr),
        str = @cfunctionOO(pyjlany_str),
        getattro = @cfunctionOOO(pyjlany_getattro),
        setattro = @cfunctionIOOO(pyjlany_setattro),
        call = @cfunctionOOOO(pyjlany_call),
        iter = @cfunctionOO(pyjlany_iter),
        richcompare = @cfunctionOOOI(pyjlany_richcompare),
        as_mapping = cacheptr!(c, fill(PyMappingMethods(
            length = @cfunctionZO(pyjlany_length),
            subscript = @cfunctionOOO(pyjlany_getitem),
            ass_subscript = @cfunctionIOOO(pyjlany_setitem),
        ))),
        as_sequence = cacheptr!(c, fill(PySequenceMethods(
            contains = @cfunctionIOO(pyjlany_contains),
        ))),
        as_number = cacheptr!(c, fill(PyNumberMethods(
            positive = @cfunctionOO(pyjlany_positive),
            negative = @cfunctionOO(pyjlany_negative),
            absolute = @cfunctionOO(pyjlany_absolute),
            power = @cfunctionOOOO(pyjlany_power),
            add = @cfunctionOOO(pyjlany_binop(+)),
            subtract = @cfunctionOOO(pyjlany_binop(-)),
            multiply = @cfunctionOOO(pyjlany_binop(*)),
            truedivide = @cfunctionOOO(pyjlany_binop(/)),
            divmod = @cfunctionOOO(pyjlany_binop((x,y) -> (fld(x,y), mod(x,y)))),
            floordivide = @cfunctionOOO(pyjlany_binop(fld)),
            remainder = @cfunctionOOO(pyjlany_binop(mod)),
            lshift = @cfunctionOOO(pyjlany_binop(<<)),
            rshift = @cfunctionOOO(pyjlany_binop(>>)),
            and = @cfunctionOOO(pyjlany_binop(&)),
            xor = @cfunctionOOO(pyjlany_binop(⊻)),
            or = @cfunctionOOO(pyjlany_binop(|)),
        ))),
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "__dir__"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlany_dir),
            ),
            PyMethodDef(
                name = cacheptr!(c, "_repr_mimebundle_"),
                flags = Py_METH_VARARGS | Py_METH_KEYWORDS,
                meth = @cfunctionOOOO(pyjlany_repr_mimebundle),
            ),
            PyMethodDef(
                name = cacheptr!(c, "_jl_raw"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlany_toraw),
            ),
            PyMethodDef(
                name = cacheptr!(c, "_jl_display"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlany_display),
            ),
            PyMethodDef(
                name = cacheptr!(c, "_jl_help"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOOO(pyjlany_help),
            ),
            PyMethodDef(),
        ]),
        getset = cacheptr!(c, [
            PyGetSetDef(
                name = cacheptr!(c, "__name__"),
                get = @cfunctionOOP(pyjlany_name),
            ),
            PyGetSetDef(),
        ])
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end
