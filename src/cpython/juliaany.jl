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
                methods = [
                    (name = "__dir__", flags = Py_METH_NOARGS, meth = pyjlany_dir),
                    (
                        name = "_repr_html_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("text/html")),
                    ),
                    (
                        name = "_repr_markdown_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("text/markdown")),
                    ),
                    (
                        name = "_repr_json_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("text/json")),
                    ),
                    (
                        name = "_repr_javascript_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("application/javascript")),
                    ),
                    (
                        name = "_repr_pdf_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("application/pdf")),
                    ),
                    (
                        name = "_repr_jpeg_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("image/jpeg")),
                    ),
                    (
                        name = "_repr_png_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("image/png")),
                    ),
                    (
                        name = "_repr_svg_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("image/svg+xml")),
                    ),
                    (
                        name = "_repr_latex_",
                        flags = Py_METH_NOARGS,
                        meth = pyjlany_repr_mime(MIME("text/latex")),
                    ),
                    (name = "__jl_raw", flags = Py_METH_NOARGS, meth = pyjlany_toraw),
                    (name = "__jl_show", flags = Py_METH_NOARGS, meth = pyjlany_show),
                    (name = "__jl_help", flags = Py_METH_NOARGS, meth = pyjlany_help),
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

struct pyjlany_repr_mime{M<:MIME}
    mime::M
end
(f::pyjlany_repr_mime{M})(xo::PyPtr, ::PyPtr) where {M} = begin
    x = PyJuliaValue_GetValue(xo)
    io = IOBuffer()
    try
        show(io, f.mime, x)
    catch err
        if err isa MethodError && err.f === show && err.args === (io, f.mime, x)
            return PyNone_New()
        else
            PyErr_SetJuliaError(err)
            return PyNULL
        end
    end
    data = take!(io)
    if istextmime(f.mime)
        PyUnicode_From(data)
    else
        PyBytes_From(data)
    end
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

pyjlany_show(xo::PyPtr, ::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)
        io = stdout
        io = IOContext(
            io,
            :limit => get(io, :limit, true),
            :compact => get(io, :limit, true),
            :color => get(io, :color, true),
        )
        show(io, MIME("text/plain"), x)
        println(io)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end

pyjlany_help(xo::PyPtr, ::PyPtr) =
    try
        x = Docs.doc(PyJuliaValue_GetValue(xo))
        io = stdout
        io = IOContext(
            io,
            :limit => get(io, :limit, true),
            :compact => get(io, :limit, true),
            :color => get(io, :color, true),
        )
        show(io, MIME("text/plain"), x)
        println(io)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyNULL
    end
