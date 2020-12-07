const PYJLGCCACHE = Dict{CPyPtr, Any}()

function cpycatch(f, ::Type{T}=CPyPtr) where {T}
    try
        cpyreturn(T, f())
    catch err
        if err isa PythonRuntimeError
            # We restore Python errors.
            # TODO: Is this the right behaviour?
            pyerrrestore(err)
        else
            # Other (Julia) errors are raised as a JuliaException
            bt = catch_backtrace()
            val = try
                pyjlbase((err, bt))
            catch
                pynone
            end
            pyerrset(pyjlexception, val)
        end
        cpyerrval(T)
    end
end

cpyreturn(::Type{T}, x::T) where {T} = x
cpyreturn(::Type{CPyPtr}, x::PyObject) = CPyPtr(pyptr(pyincref!(x)))
cpyreturn(::Type{T}, x::Number) where {T<:Number} = convert(T, x)
cpyreturn(::Type{T}, x::Ptr) where {T<:Ptr} = T(x)

cpyerrval(::Type{Nothing}) = nothing
cpyerrval(::Type{T}) where {T<:Number} = zero(T)-one(T)
cpyerrval(::Type{T}) where {T<:Ptr} = T(C_NULL)

### jlexception

const pyjlexception = pylazyobject() do
    # make the type
    c = []
    t = cpynewtype!(c; name="julia.Exception", base=pyexception, basicsize=0)
    # put into a 0-dim array and take a pointer
    ta = fill(t)
    ptr = pointer(ta)
    # ready the type
    check(C.PyType_Ready(ptr))
    # success
    PYJLGCCACHE[CPyPtr(ptr)] = push!(c, ta)
    pyborrowedobject(ptr)
end
export pyjlexception

### jlfunction

@kwdef struct CPyJlFunctionObject
    ob_base :: C.PyObject = C.PyObject()
    call :: Ptr{Cvoid} = C_NULL
end

cpyjlfunction_call(o::CPyPtr, args::CPyPtr, kwargs::CPyPtr) = ccall(UnsafePtr{CPyJlFunctionObject}(o).call[!], CPyPtr, (CPyPtr, CPyPtr), args, kwargs)
function cpyjlfunction_dealloc(o::CPyPtr)
    delete!(PYJLGCCACHE, o)
    ccall(UnsafePtr{C.PyTypeObject}(C.Py_Type(o)).free[!], Cvoid, (CPyPtr,), o)
    nothing
end

const pyjlfunctiontype = pylazyobject() do
    # make the type
    c = []
    t = cpynewtype!(c;
        name = "julia.Function",
        base = pyobjecttype,
        basicsize = sizeof(CPyJlFunctionObject),
        flags = C.Py_TPFLAGS_HAVE_VERSION_TAG | (CONFIG.isstackless ? C.Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
        call = @cfunction(cpyjlfunction_call, CPyPtr, (CPyPtr, CPyPtr, CPyPtr)),
        dealloc = @cfunction(cpyjlfunction_dealloc, Cvoid, (CPyPtr,)),
    )
    # put into a 0-dim array and take a pointer
    ta = fill(t)
    ptr = pointer(ta)
    # ready the type
    check(C.PyType_Ready(ptr))
    # success
    PYJLGCCACHE[CPyPtr(ptr)] = push!(c, ta)
    pyborrowedobject(ptr)
end
export pyjlfunctiontype

struct cpyjlfunction_call_impl{F}
    f :: F
end
(f::cpyjlfunction_call_impl)(_args::CPyPtr, _kwargs::CPyPtr) = cpycatch() do
    if _kwargs != C_NULL
        args = pyborrowedobject(_args)
        kwargs = Dict{Symbol,PyObject}(Symbol(string(k)) => v for (k,v) in pyborrowedobject(_kwargs).items())
        pyobject(f.f(args...; kwargs...))
    elseif _args != C_NULL
        args = pyborrowedobject(_args)
        nargs = length(args)
        if nargs == 0
            pyobject(f.f())
        elseif nargs == 1
            pyobject(f.f(args[0]))
        elseif nargs == 2
            pyobject(f.f(args[0], args[1]))
        elseif nargs == 3
            pyobject(f.f(args[0], args[1], args[2]))
        else
            pyobject(f.f(args...))
        end
    else
        pyobject(f.f())
    end
end

function pyjlfunction(f)
    # allocate an object
    o = check(C._PyObject_New(pyjlfunctiontype))
    # set value
    p = UnsafePtr{CPyJlFunctionObject}(pyptr(o))
    c = []
    p.call[] = cacheptr!(c, @cfunction($(cpyjlfunction_call_impl(f)), CPyPtr, (CPyPtr, CPyPtr)))
    PYJLGCCACHE[pyptr(o)] = c
    # done
    return o
end
export pyjlfunction

### jlbasevalue

@kwdef struct CPyJlValueObject{T}
    ob_base :: C.PyObject = C.PyObject()
    value :: Ptr{Cvoid} = C_NULL
    weaklist :: CPyPtr = C_NULL
end
const CPyJlPtr{T} = Ptr{CPyJlValueObject{T}}

function cpyjlvalue_dealloc(o::CPyPtr)
    delete!(PYJLGCCACHE, o)
    UnsafePtr{CPyJlValueObject{Any}}(o).weaklist[!] == C_NULL || C.PyObject_ClearWeakRefs(o)
    ccall(UnsafePtr{C.PyTypeObject}(C.Py_Type(o)).free[!], Cvoid, (CPyPtr,), o)
    nothing
end

function cpyjlvalue_new(t::CPyPtr, args::CPyPtr, kwargs::CPyPtr)
    o = ccall(UnsafePtr{C.PyTypeObject}(t).alloc[!], CPyPtr, (CPyPtr, C.Py_ssize_t), t, 0)
    if o != C_NULL
        p = UnsafePtr{CPyJlValueObject{Any}}(o)
        p.value[] = C_NULL
        p.weaklist[] = C_NULL
    end
    o
end

cpyjlvalue_get_buffer(_o::CPyPtr, buf::Ptr{C.Py_buffer}, flags::Cint) = cpycatch(Cint) do
    o = pyborrowedobject(_o)
    t = pytype(o)
    if pyhasattr(t, "__jl_enable_buffer__") && pytruth(t.__jl_enable_buffer__)
        pyjl_get_buffer(o, buf, flags)
    else
        pythrow(pybuffererror("Buffer protocol not supported by '$(t.__name__)'"))
    end
end

function cpyjlvalue_release_buffer(_o::CPyPtr, buf::Ptr{C.Py_buffer})
    delete!(PYJLBUFCACHE, UnsafePtr(buf).internal[])
    nothing
end

const pyjlbasetype = pylazyobject() do
    # make the type
    c = []
    t = cpynewtype!(c;
        name = "julia.ValueBase",
        basicsize = sizeof(CPyJlValueObject{Any}),
        new = @cfunction(cpyjlvalue_new, CPyPtr, (CPyPtr, CPyPtr, CPyPtr)),
        dealloc = @cfunction(cpyjlvalue_dealloc, Cvoid, (CPyPtr,)),
        flags = C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG | (CONFIG.isstackless ? C.Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
        weaklistoffset = fieldoffset(CPyJlValueObject{Any}, 3),
        getattro = C.@pyglobal(:PyObject_GenericGetAttr),
        setattro = C.@pyglobal(:PyObject_GenericSetAttr),
        doc = "A Julia value with no semantics.",
        as_buffer = (get=@cfunction(cpyjlvalue_get_buffer, Cint, (CPyPtr, Ptr{C.Py_buffer}, Cint)), release=@cfunction(cpyjlvalue_release_buffer, Cvoid, (CPyPtr, Ptr{C.Py_buffer})))
    )
    # put into a 0-dim array and take a pointer
    ta = fill(t)
    ptr = pointer(ta)
    # ready the type
    check(C.PyType_Ready(ptr))
    # success
    PYJLGCCACHE[CPyPtr(ptr)] = push!(c, ta)
    pyborrowedobject(ptr)
end
export pyjlbasetype

function pyjlnewvalue(x, t::PyObject)
    pyissubclass(t, pyjlbasetype) || error("Expecting a subtype of 'julia.ValueBase'")
    # allocate
    o = t()
    # set value
    p = UnsafePtr{CPyJlValueObject{Any}}(pyptr(o))
    p.value[], PYJLGCCACHE[pyptr(o)] = pointer_from_obj(x)
    # done
    return o
end

pyjlbase(x) = pyjlnewvalue(x, pyjlbasetype)
export pyjlbase

pyisjl(o::PyObject) = pyisinstance(o, pyjlbasetype)
export pyisjl

cpyjlgetvalue(o::CPyJlPtr{T}) where {T} = Base.unsafe_pointer_to_objref(UnsafePtr(o).value[!]) :: T
cpyjlgetvalue(o::Ptr) = cpyjlgetvalue(CPyJlPtr{Any}(o))

function pyjlgetvalue(o::PyObject, ::Type{T}=Any) where {T}
    pyisinstance(o, pyjlbasetype) || pythrow(pytypeerror("Expecting a Python 'julia.ValueBase'"))
    ptr = UnsafePtr(CPyJlPtr{Any}(pyptr(o))).value[!]
    ptr == C_NULL && pythrow(pyvalueerror("Value is NULL"))
    Base.unsafe_pointer_to_objref(ptr)::T
end
export pyjlgetvalue

### jlrawvalue

const PYJLRAWTYPES = Dict{Type,PyObject}()

pyjlrawtype(::Type{T}) where {T} = get!(PYJLRAWTYPES, T) do
    name = "julia.RawValue[$T]"
    base = T==Any ? pyjlbasetype : pyjlrawtype(T==DataType ? Type : supertype(T))
    attrs = pydict()
    attrs["__slots__"] = pytuple()
    attrs["__repr__"] = pymethod(o -> "<jl $(repr(pyjlgetvalue(o, T)))>")
    attrs["__str__"] = pymethod(o -> string(pyjlgetvalue(o, T)))
    attrs["__call__"] = pymethod((o, args...; kwargs...) -> pyjlraw(pyjlgetvalue(o, T)(map(pyjlgetvalue, args)...)))
    attrs["__doc__"] = """
    A Julia '$T' with basic Julia semantics.
    """
    # Logic
    # Note that comparisons use isequal and isless, so that hashing is supported
    attrs["__eq__"] = pymethod((o, x) -> pybool(isequal(pyjlgetvalue(o, T), pyjlgetvalue(x))))
    attrs["__lt__"] = pymethod((o, x) -> pybool(isless(pyjlgetvalue(o, T), pyjlgetvalue(x))))
    attrs["__bool__"] = pymethod(o -> (v=pyjlgetvalue(o, T); v isa Bool ? pybool(v) : pythrow(pytypeerror("Only 'Bool' can be tested for truthyness"))))
    attrs["__hash__"] = pymethod(o -> pyint(hash(pyjlgetvalue(o, T))))
    # Containers
    attrs["__contains__"] = pymethod((o, v) -> pybool(pyjlgetvalue(v) in pyjlgetvalue(o, T)))
    attrs["__getitem__"] = pymethod((o, i) -> pyjlraw(getindex(pyjlgetvalue(o, T), (pyistuple(i) ? [pyjlgetvalue(j) for j in i] : [pyjlgetvalue(i)])...)))
    attrs["__setitem__"] = pymethod((o, i, v) -> pyjlraw(setindex!(pyjlgetvalue(o, T), pyjlgetvalue(v), (pyistuple(i) ? [pyjlgetvalue(j) for j in i] : [pyjlgetvalue(i)])...)))
    attrs["__iter__"] = pymethod(o -> pyjlraw(Iterator(pyjlgetvalue(o, T), nothing)))
    if T <: Iterator
        attrs["__next__"] = pymethod(_o -> begin
            o = pyjlgetvalue(_o, T)
            if o.st === nothing
                it = iterate(o.val)
            else
                it = iterate(o.val, something(o.st))
            end
            if it === nothing
                pythrow(pystopiteration())
            else
                x, st = it
                o.st = Some(st)
                return pyjlraw(x)
            end
        end)
    end
    # Arithmetic
    attrs["__add__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) + pyjlgetvalue(x)))
    attrs["__sub__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) - pyjlgetvalue(x)))
    attrs["__mul__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) * pyjlgetvalue(x)))
    attrs["__truediv__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) / pyjlgetvalue(x)))
    attrs["__floordiv__"] = pymethod((o, x) -> pyjlraw(fld(pyjlgetvalue(o, T), pyjlgetvalue(x))))
    attrs["__mod__"] = pymethod((o, x) -> pyjlraw(mod(pyjlgetvalue(o, T), pyjlgetvalue(x))))
    attrs["__pow__"] = pymethod((o, x, m=pynone) -> pyjlraw(pyisnone(m) ? pyjlgetvalue(o, T) ^ pyjlgetvalue(x) : powermod(pyjlgetvalue(o, T), pyjlgetvalue(x), pyjlgetvalue(m))))
    attrs["__lshift__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) << pyjlgetvalue(x)))
    attrs["__rshift__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) >> pyjlgetvalue(x)))
    attrs["__and__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) & pyjlgetvalue(x)))
    attrs["__xor__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) ⊻ pyjlgetvalue(x)))
    attrs["__or__"] = pymethod((o, x) -> pyjlraw(pyjlgetvalue(o, T) | pyjlgetvalue(x)))
    attrs["__neg__"] = pymethod(o -> pyjlraw(-pyjlgetvalue(o, T)))
    attrs["__pos__"] = pymethod(o -> pyjlraw(+pyjlgetvalue(o, T)))
    attrs["__abs__"] = pymethod(o -> pyjlraw(abs(pyjlgetvalue(o, T))))
    attrs["__invert__"] = pymethod(o -> pyjlraw(~pyjlgetvalue(o, T)))
    attrs["__complex__"] = pymethod(o -> pycomplex(convert(Complex{Cdouble}, pyjlgetvalue(o, T))))
    attrs["__float__"] = pymethod(o -> pyfloat(convert(Cdouble, pyjlgetvalue(o, T))))
    attrs["__int__"] = attrs["__index__"] = pymethod(o -> pyint(convert(Integer, pyjlgetvalue(o, T))))
    # Julia-specific
    attrs["getfield"] = pymethod((o, k) -> pyjlraw(getfield(pyjlgetvalue(o, T), pyjlraw_propertyname(k))))
    attrs["getprop"] = pymethod((o, k) -> pyjlraw(getproperty(pyjlgetvalue(o, T), pyjlraw_propertyname(k))))
    attrs["setfield"] = pymethod((o, k, v) -> (setfield!(pyjlgetvalue(o, T), pyjlraw_propertyname(k), pyjlgetvalue(v)); nothing))
    attrs["setprop"] = pymethod((o, k, v) -> (setproperty!(pyjlgetvalue(o, T), pyjlraw_propertyname(k), pyjlgetvalue(v)); nothing))
    attrs["prop"] = pymethod((o, k, v=nothing) -> v===nothing ? o.getprop(k) : o.setprop(k, v))
    attrs["field"] = pymethod((o, k, v=nothing) -> v===nothing ? o.getfield(k) : o.setfield(k, v))
    attrs["typeof"] = isconcretetype(T) ? pymethod(o -> pyjlraw(T)) : pymethod(o -> pyjlraw(typeofpyjlgetvalue(o, T)))
    # Done
    pytypetype(name, (base,), attrs)
end
export pyjlrawtype

pyjlraw_propertyname(k::PyObject) = pyisstr(k) ? Symbol(pystr_asjuliastring(k)) : pyisint(k) ? pyint_tryconvert(Int, k) : pyjlgetvalue(k)

pyjlraw(x, ::Type{T}) where {T} = x isa T ? pyjlnewvalue(x, pyjlrawtype(T)) : error("expecting a `$T`")
pyjlraw(x) = pyjlraw(x, typeof(x))
export pyjlraw

### jlvalue

const PYJLTYPES = Dict{Type,PyObject}()

function pyjl_supertypes(::Type{T}) where {T}
    r = []
    S = T
    while S !== nothing
        push!(r, S)
        S = pyjl_supertype(S)
    end
    r
end

pyjl_supertype(::Type{T}) where {T} = supertype(T)
pyjl_supertype(::Type{Any}) = nothing
pyjl_supertype(::Type{DataType}) = Type

pyjl_valuetype(::Type{T}) where {T} = T

abstract type PyJlSubclass{S} end

pyjl_supertype(::Type{T}) where {S, T<:PyJlSubclass{S}} =
    if supertype(T) <: PyJlSubclass
        if supertype(supertype(T)) <: PyJlSubclass
            supertype(T)
        else
            S
        end
    else
        error("cannot instantiate `PyJlSubclass`")
    end
pyjl_supertype(::Type{T}) where {T<:PyJlSubclass} = error()

pyjl_valuetype(::Type{T}) where {S, T<:PyJlSubclass{S}} = pyjl_valuetype(S)
pyjl_valuetype(::Type{T}) where {T<:PyJlSubclass} = error()

pyjltype(::Type{T}) where {T} = get!(PYJLTYPES, T) do
    # Name
    name = "julia.Value[$T]"

    # Bases
    Ss = pyjl_supertypes(T)
    bases = [length(Ss) == 1 ? pyjlbasetype : pyjltype(Ss[2])]
    mixin = pyjl_mixin(T)
    pyisnone(mixin) || push!(bases, mixin)

    # Attributes
    V = pyjl_valuetype(T)
    attrs = Dict{String,PyObject}()
    attrs["__slots__"] = pytuple()
    for S in reverse(Ss)
        V <: pyjl_valuetype(S) || error()
        pyjl_addattrs(attrs, S, V)
    end

    # Make the type
    t = pytypetype(name, pytuple_fromiter(bases), pydict_fromstringiter(attrs))

    # Register an abstract base class
    abc = pyjl_abc(T)
    pyisnone(abc) || abc.register(t)

    # Done
    return t
end
export pyjltype

pyjl(x, ::Type{T}) where {T} = x isa pyjl_valuetype(T) ? pyjlnewvalue(x, pyjltype(T)) : error("expecting a `$(pyjl_valuetype(T))`")
pyjl(x) = pyjl(x, typeof(x))
export pyjl

### Any

pyjl_abc(::Type) = pynone
pyjl_mixin(::Type) = pynone

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Any, V<:T}
    t["__repr__"] = pymethod(o -> "<jl $(repr(pyjlgetvalue(o, V)))>")
    t["__str__"] = pymethod(o -> string(pyjlgetvalue(o, V)))
    t["__doc__"] = """
    A Julia '$T' with Python semantics.
    """
    if hasmethod(length, Tuple{V})
        t["__len__"] = pymethod(o -> length(pyjlgetvalue(o, V)))
    end
    if hasmethod(hash, Tuple{V})
        t["__hash__"] = pymethod(o -> hash(pyjlgetvalue(o, V)))
    end
    if hasmethod(in, Tuple{Union{}, V})
        t["__contains__"] = pymethod((_o, _x) -> begin
            o = pyjlgetvalue(_o, V)
            x = pytryconvert_element(o, _x)
            x === PyConvertFail() ? false : in(x, o)
        end)
    end
    if hasmethod(reverse, Tuple{V})
        t["__reversed__"] = pymethod(o -> pyjl(reverse(pyjlgetvalue(o, V))))
    end
    if hasmethod(iterate, Tuple{V})
        t["__iter__"] = pymethod(o -> pyjl(Iterator(pyjlgetvalue(o, V), nothing)))
    end
    t["__call__"] = pymethod((o, args...; kwargs...) -> pyobject(pyjlgetvalue(o, V)(args...; kwargs...)))
    if hasmethod(getindex, Tuple{V, Union{}})
        t["__getitem__"] = pymethod((_o, _k) -> begin
            o = pyjlgetvalue(_o, V)
            k = pytryconvert_indices(o, _k)
            if k === PyConvertFail()
                pythrow(pytypeerror("invalid index"))
            end
            pyobject(o[k...])
        end)
    end
    if hasmethod(setindex!, Tuple{V, Union{}, Union{}})
        t["__setitem__"] = pymethod((_o, _k, _v) -> begin
            o = pyjlgetvalue(_o, V)
            k = pytryconvert_indices(o, _k)
            k === PyConvertFail && pythrow(pytypeerror("invalid index"))
            v = pytryconvert_value(o, _v, k...)
            v === PyConvertFail && pythrow(pytypeerror("invalid value"))
            o[k...] = v
            pynone
        end)
    end
    t["__dir__"] = pymethod(o -> begin
        d = pyobjecttype.__dir__(o)
        d.extend(pylist([pyjl_attrname_jl2py(string(k)) for k in propertynames(pyjlgetvalue(o, V))]))
        d
    end)
    t["__getattr__"] = pymethod((_o, _k) -> begin
        # first do a generic lookup
        _x = C.PyObject_GenericGetAttr(_o, _k)
        if _x != C_NULL
            return pynewobject(_x)
        elseif !pyerroccurred(pyattributeerror)
            pythrow()
        end
        st = pyerrfetch()
        # try to get a julia property with this name
        o = pyjlgetvalue(_o, V)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(_k)))
        if hasproperty(o, k)
            return pyobject(getproperty(o, k))
        end
        throw(PythonRuntimeError(st...))
    end)
    t["__setattr__"] = pymethod((_o, _k, _v) -> begin
        # first do a generic lookup
        _x = C.PyObject_GenericSetAttr(_o, _k, _v)
        if _x != -1
            return pynone
        elseif !pyerroccurred(pyattributeerror)
            pythrow()
        end
        st = pyerrfetch()
        # try to set a julia property with this name
        o = pyjlgetvalue(_o, V)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(_k)))
        if hasproperty(o, k)
            v = pyconvert(Any, _v)
            setproperty!(o, k, v)
            return pynone
        end
        throw(PythonRuntimeError(st...))
    end)
    # Blindly try to convert _x to something compatible with x
    prom(o, _x) = if pyisjl(_x)
        return pyjlgetvalue(_x)
    else
        x = pytryconvert(typeof(o), _x)
        x === PyConvertFail() ? pyconvert(Any, x) : x
    end
    binop(op) = pymethod((_o, _x) -> begin
        try
            o = pyjlgetvalue(_o, V)
            x = prom(o, _x)
            pyobject(op(o, x))
        catch
            pynotimplemented
        end
    end)
    hasmethod(+, Tuple{V, Union{}}) && (t["__add__"] = binop(+))
    hasmethod(-, Tuple{V, Union{}}) && (t["__sub__"] = binop(-))
    hasmethod(*, Tuple{V, Union{}}) && (t["__mul__"] = binop(*))
    hasmethod(/, Tuple{V, Union{}}) && (t["__truediv__"] = binop(/))
    hasmethod(fld, Tuple{V, Union{}}) && (t["__floordiv__"] = binop(fld))
    hasmethod(mod, Tuple{V, Union{}}) && (t["__mod__"] = binop(mod))
    hasmethod(<<, Tuple{V, Union{}}) && (t["__lshift__"] = binop(<<))
    hasmethod(>>, Tuple{V, Union{}}) && (t["__rshift__"] = binop(>>))
    hasmethod(&, Tuple{V, Union{}}) && (t["__and__"] = binop(&))
    hasmethod(|, Tuple{V, Union{}}) && (t["__or__"] = binop(|))
    hasmethod(⊻, Tuple{V, Union{}}) && (t["__xor__"] = binop(⊻))
    if hasmethod(^, Tuple{V, Union{}})
        t["__pow__"] = pymethod((_o, _x, _m=pynone) -> begin
            try
                o = pyjlgetvalue(_o, V)
                x = prom(o, _x)
                if pyisnone(_m)
                    pyobject(o^x)
                else
                    m = prom(o, _m)
                    pyobject(powermod(o, x, m))
                end
            catch
                pynotimplemented
            end
        end)
    end
    hasmethod(==, Tuple{V, Union{}}) && (t["__eq__"] = binop(==))
    hasmethod(!=, Tuple{V, Union{}}) && (t["__ne__"] = binop(!=))
    hasmethod(<=, Tuple{V, Union{}}) && (t["__le__"] = binop(<=))
    hasmethod(< , Tuple{V, Union{}}) && (t["__lt__"] = binop(< ))
    hasmethod(>=, Tuple{V, Union{}}) && (t["__ge__"] = binop(>=))
    hasmethod(> , Tuple{V, Union{}}) && (t["__gt__"] = binop(> ))
end

pyjl_attrname_py2jl(x::AbstractString) =
    replace(x, r"_[b]+$" => s -> "!"^(length(s)-1))

pyjl_attrname_jl2py(x::AbstractString) =
    replace(x, r"!+$" => s -> "_" * "b"^(length(s)))

### Nothing & Missing (falsy)

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Union{Nothing,Missing}, V<:T}
    t["__bool__"] = pymethod(o -> pyfalse)
end

### Iterator (as Iterator)

mutable struct Iterator{T}
    val :: T
    st :: Union{Nothing, Some}
end

pyjl_mixin(::Type{T}) where {T<:Iterator} = pyiteratorabc

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Iterator, V<:T}
    t["__iter__"] = pymethod(o -> o)
    t["__next__"] = pymethod(_o -> begin
        o = pyjlgetvalue(_o, V)
        if o.st === nothing
            it = iterate(o.val)
        else
            it = iterate(o.val, something(o.st))
        end
        if it === nothing
            pythrow(pystopiteration())
        else
            x, st = it
            o.st = Some(st)
            return pyobject(x)
        end
    end)
end

### Number

pyjl_mixin(::Type{T}) where {T<:Number} = pynumbersmodule.Number
pyjl_mixin(::Type{T}) where {T<:Complex} = pynumbersmodule.Complex
pyjl_mixin(::Type{T}) where {T<:Real} = pynumbersmodule.Real
pyjl_mixin(::Type{T}) where {T<:Rational} = pynumbersmodule.Rational
pyjl_mixin(::Type{T}) where {T<:Integer} = pynumbersmodule.Integral

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Number, V<:T}
    t["__bool__"] = pymethod(o -> pybool(!iszero(pyjlgetvalue(o, V))))
    t["__pos__"] = pymethod(o -> pyobject(+pyjlgetvalue(o, V)))
    t["__neg__"] = pymethod(o -> pyobject(-pyjlgetvalue(o, V)))
end

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Union{Complex,Real}, V<:T}
    t["real"] = pyproperty(o -> pyobject(real(pyjlgetvalue(o, V))))
    t["imag"] = pyproperty(o -> pyobject(imag(pyjlgetvalue(o, V))))
    t["conjugate"] = pymethod(o -> pyobject(conj(pyjlgetvalue(o, V))))
    t["__abs__"] = pymethod(o -> pyobject(abs(pyjlgetvalue(o, V))))
    t["__complex__"] = pymethod(o -> pycomplex(pyjlgetvalue(o, V)))
    if V<:Real
        t["__float__"] = pymethod(o -> pyfloat(pyjlgetvalue(o, V)))
        t["__trunc__"] = pymethod(o -> pyint(trunc(BigInt, pyjlgetvalue(o, V))))
        t["__round__"] = pymethod((o,n=pynone) -> pyisnone(n) ? pyint(round(BigInt, pyjlgetvalue(o, V))) : pyjl(round(pyjlgetvalue(o, V), digits=pyconvert(Int, n))))
        t["__floor__"] = pymethod(o -> pyint(floor(BigInt, pyjlgetvalue(o, V))))
        t["__ceil__"] = pymethod(o -> pyint(ceil(BigInt, pyjlgetvalue(o, V))))
    end
end

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Union{Integer,Rational}, V<:T}
    t["numerator"] = pyproperty(o -> pyobject(numerator(pyjlgetvalue(o, V))))
    t["denominator"] = pyproperty(o -> pyobject(denominator(pyjlgetvalue(o, V))))
    if V<:Integer
        t["__int__"] = pymethod(o -> pyint(pyjlgetvalue(o, V)))
        t["__invert__"] = pymethod(o -> pyobject(~pyjlgetvalue(o, V)))
    end
end

### Dict (as Mapping)

pyjl_mixin(::Type{T}) where {T<:AbstractDict} = pymutablemappingabc

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:AbstractDict, V<:T}
    t["__iter__"] = pymethod(o -> pyjl(Iterator(keys(pyjlgetvalue(o, V)), nothing)))
    t["__getitem__"] = pymethod((_o, _k) -> begin
        o = pyjlgetvalue(_o, V)
        k = pytryconvert(keytype(o), _k)
        if k === PyConvertFail() || !haskey(o, k)
            pythrow(pykeyerror(_k))
        end
        pyobject(o[k])
    end)
    t["__setitem__"] = pymethod((_o, _k, _v) -> begin
        o = pyjlgetvalue(_o, V)
        k = pytryconvert(keytype(o), _k)
        if k === PyConvertFail()
            pythrow(pykeyerror(_k))
        end
        v = pytryconvert(valtype(o), _v)
        v === PyConvertFail() && pythrow(pytypeerror("invalid value of type '$(pytype(_v).__name__)'"))
        o[k] = v
        pynone
    end)
    t["__delitem__"] = pymethod((_o, _k) -> begin
        o = pyjlgetvalue(_o, V)
        k = pytryconvert(keytype(o), _k)
        if k === PyConvertFail() || !haskey(o, k)
            pythrow(pykeyerror(_k))
        end
        delete!(o, k)
        pynone
    end)
    t["__contains__"] = pymethod((_o, _k) -> begin
        o = pyjlgetvalue(_o, V)
        k = pytryconvert(keytype(o), _k)
        k === PyConvertFail() ? false : haskey(o, k)
    end)
    t["clear"] = pymethod(o -> (empty!(pyjlgetvalue(o, V)); pynone))
end

### Set (as Set)

pyjl_mixin(::Type{T}) where {T<:AbstractSet} = pymutablesetabc

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:AbstractSet, V<:T}
    t["add"] = pymethod((_o, _v) -> begin
        o = pyjlgetvalue(_o, V)
        v = pytryconvert(eltype(o), _v)
        v === PyConvertFail() && pythrow(pytypeerror("Invalid value of type '$(pytype(v).__name__)'"))
        push!(o, v)
        pynone
    end)
    t["discard"] = pymethod((_o, _v) -> begin
        o = pyjlgetvalue(_o, V)
        v = pytryconvert(eltype(o), _v)
        v === PyConvertFail() || delete!(o, v)
        pynone
    end)
    t["clear"] = pymethod(o -> (empty!(pyjlgetvalue(o, V)); pynone))
end

### Array (as Collection)
### Vector (as Sequence)

pyjl_mixin(::Type{T}) where {T<:AbstractArray} = pycollectionabc
pyjl_mixin(::Type{T}) where {T<:AbstractVector} = pymutablesequenceabc

function pyjl_axisidx(ax, k::PyObject)
    # slice
    if pyisslice(k)
        error("slicing not implemented")
    end
    # convert to int
    if pyisint(k)
        # nothing to do
    elseif pyhasattr(k, "__index__")
        k = k.__index__()
    else
        pythrow(pytypeerror("Indices must be 'int' or 'slice', not '$(pytype(k).__name__)'"))
    end
    # convert to julia int
    i = check(C.PyLong_AsLongLong(k), true)
    # negative indexing
    j = i<0 ? i+length(ax) : i
    # bounds check
    0 ≤ j < length(ax) || pythrow(pyindexerror("Index out of range"))
    # adjust for zero-up indexing
    j + first(ax)
end

function pyjl_arrayidxs(o::AbstractArray{T,N}, k::PyObject) where {T,N}
    if pyistuple(k)
        N == pylen(k) && return ntuple(i -> pyjl_axisidx(axes(o, i), k[i-1]), N)
    else
        N == 1 && return (pyjl_axisidx(axes(o, 1), k),)
    end
    pythrow(pytypeerror("Expecting a tuple of $N indices"))
end

pyjl_isbufferabletype(::Type{T}) where {T} =
    T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Bool, Ptr{Cvoid})
pyjl_isbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) && Base.allocatedinline(T) && all(pyjl_isbufferabletype, fieldtypes(T))
pyjl_isbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjl_isbufferabletype(T)

pyjl_isarrayabletype(::Type{T}) where {T} =
    T in (UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Bool, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64})

islittleendian() = Base.ENDIAN_BOM == 0x04030201 ? true : Base.ENDIAN_BOM == 0x01020304 ? false : error()

function pytypestrformat(::Type{T}) where {T}
    c = islittleendian() ? '<' : '>'
    T ==    Int8 ? ("$(c)i1", pynone) :
    T ==   UInt8 ? ("$(c)u1", pynone) :
    T ==   Int16 ? ("$(c)i2", pynone) :
    T ==  UInt16 ? ("$(c)u2", pynone) :
    T ==   Int32 ? ("$(c)i4", pynone) :
    T ==  UInt32 ? ("$(c)u4", pynone) :
    T ==   Int64 ? ("$(c)i8", pynone) :
    T ==  UInt64 ? ("$(c)u8", pynone) :
    T == Float16 ? ("$(c)f2", pynone) :
    T == Float32 ? ("$(c)f4", pynone) :
    T == Float64 ? ("$(c)f8", pynone) :
    error("not implemented")
end

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:AbstractArray, V<:T}
    if hasmethod(pointer, Tuple{V}) && hasmethod(strides, Tuple{V})
        if pyjl_isbufferabletype(eltype(V))
            t["__jl_enable_buffer__"] = true
        end
        if pyjl_isarrayabletype(eltype(V))
            t["__array_interface__"] = pyproperty(_o -> begin
                o = pyjlgetvalue(_o, V)
                typestr, descr = pytypestrformat(eltype(o))
                pydict(
                    shape = size(o),
                    typestr = typestr,
                    descr = descr,
                    data = (UInt(pointer(o)), !ismutablearray(o)),
                    strides = strides(o) .* Base.aligned_sizeof(eltype(o)),
                    version = 3,
                )
            end)
        end
    end
    if V <: PyObjectArray
        t["__array_interface__"] = pyproperty(_o -> begin
            o = pyjlgetvalue(_o, V)
            pydict(
                shape = size(o),
                typestr = "O",
                data = (UInt(pointer(o.ptrs)), false),
                strides = strides(o.ptrs) .* sizeof(CPyPtr),
                version = 3,
            )
        end)
    end
    t["ndim"] = pyproperty(o -> pyint(ndims(pyjlgetvalue(o, V))))
    t["shape"] = pyproperty(o -> pytuple(map(pyint, size(pyjlgetvalue(o, V)))))
    t["__array__"] = pymethod(o -> pyhasattr(o, "__array_interface__") ? pynumpy.asarray(o) : pynumpy.asarray(PyObjectArray(pyjlgetvalue(o, V))))
    t["__getitem__"] = pymethod((_o, _k) -> begin
        o = pyjlgetvalue(_o, V)
        k = pyjl_arrayidxs(o, _k)
        pyobject(o[k...])
    end)
    t["__setitem__"] = pymethod((_o, _k, _v) -> begin
        o = pyjlgetvalue(_o, V)
        k = pyjl_arrayidxs(o, _k)
        v = pytryconvert(eltype(o), _v)
        v === PyConvertFail() && pythrow(pytypeerror("Cannot assign value of type '$(pytype(_v).__name__)'"))
        o[k...] = v
        pynone
    end)
    t["copy"] = pymethod(o -> pyjl(copy(pyjlgetvalue(o, V))))
    if V <: AbstractVector
        t["__delitem__"] = pymethod((_o, _k) -> begin
            o = pyjlgetvalue(_o, V)
            k = pyjl_arrayidxs(o, _k)
            deleteat!(o, k...)
            pynone
        end)
        t["insert"] = pymethod((_o, _k, _v) -> begin
            o = pyjlgetvalue(_o, V)
            ax = axes(o, 1)
            k = pyjl_axisidx(first(ax):(last(ax)+1), _k)
            v = pytryconvert(eltype(o), _v)
            v === PyConvertFail() && pythrow(pytypeerror("Cannot assign value of type '$(pytype(_v).__name__)'"))
            insert!(o, k, v)
            pynone
        end)
        t["sort"] = pymethod((_o; reverse=pyfalse, key=pynone) -> begin
            o = pyjlgetvalue(_o)
            rev = pytruth(reverse)
            by = pyisnone(key) ? identity : key
            sort!(o, rev=rev, by=by)
            pynone
        end)
        t["clear"] = pymethod(o -> (empty!(pyjlgetvalue(o, V)); pynone))
    end
end

### Tuple & NamedTuple (as Sequence)

pyjl_mixin(::Type{T}) where {T<:Union{Tuple,NamedTuple}} = pysequenceabc

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:Union{Tuple,NamedTuple}, V<:T}
    t["__getitem__"] = pymethod((_o, _k) -> begin
        o = pyjlgetvalue(_o, V)
        if o isa NamedTuple && pyisstr(_k)
            k = Symbol(string(_k))
            haskey(o, k) || pythrow(pykeyerror(_k))
        else
            k = pyjl_axisidx(1:length(o), _k)
        end
        pyobject(o[k])
    end)
end

### IO (as IOBase)

pyjl_abc(::Type{T}) where {T<:IO} = pyiomodule.IOBase

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:IO, V}
    t["close"] = if hasmethod(close, Tuple{V})
        pymethod(o -> (close(pyjlgetvalue(o, V)); pynone))
    else
        pymethod(o -> pythrow(pyiounsupportedoperation("close")))
    end
    t["closed"] = if hasmethod(isopen, Tuple{V})
        pyproperty(o -> pybool(!isopen(pyjlgetvalue(o, V))))
    else
        pyproperty(o -> pythrow(pyiounsupportedoperation("closed")))
    end
    t["fileno"] = if hasmethod(fd, Tuple{V})
        pymethod(o -> pyint(fd(pyjlgetvalue(o, V))))
    else
        pymethod(o -> pythrow(pyiounsupportedoperation("fileno")))
    end
    t["flush"] = pymethod(o -> (flush(pyjlgetvalue(o, V)); pynone))
    t["isatty"] = V<:Base.TTY ? pymethod(o -> pybool(true)) : pymethod(o -> pybool(false))
    t["readable"] = if hasmethod(isreadable, Tuple{V})
        pymethod(o -> pybool(isreadable(pyjlgetvalue(o, V))))
    else
        pymethod(o -> pythrow(pyiounsupportedoperation("readable")))
    end
    t["writable"] = if hasmethod(iswritable, Tuple{V})
        pymethod(o -> pybool(iswritable(pyjlgetvalue(o, V))))
    else
        pymethod(o -> pythrow(pyiounsupportedoperation("writable")))
    end
    t["tell"] = if hasmethod(position, Tuple{V})
        pymethod(o -> pyint(position(pyjlgetvalue(o, V))))
    else
        pymethod(o -> pythrow(pyiounsupportedoperation("tell")))
    end
    t["writelines"] = pymethod((o, lines) -> begin
        wr = o.write
        for line in lines
            wr(line)
        end
        pynone
    end)
    t["seekable"] = if hasmethod(position, Tuple{V}) && hasmethod(seek, Tuple{V,Int}) && hasmethod(truncate, Tuple{V,Int})
        pymethod(o -> pybool(true))
    else
        pymethod(o -> pybool(false))
    end
    t["truncate"] = if hasmethod(truncate, Tuple{V, Int})
        pymethod((_o, _n=pynone) -> begin
            o = pyjlgetvalue(_o, V)
            n = pyisnone(_n) ? position(o) : pyconvert(Int, _n)
            truncate(o, n)
            pyint(n)
        end)
    else
        pymethod((_o, _n=pynone) -> pythrow(pyiounsupportedoperation("truncate")))
    end
    t["seek"] = if hasmethod(seek, Tuple{V, Int}) && hasmethod(position, Tuple{V})
        pymethod((_o, offset, whence=pyint(0)) -> begin
            o = pyjlgetvalue(_o, V)
            n = pyconvert(Int, offset)
            w = pyconvert(Int, whence)
            if w == 0
                seek(o, n)
            elseif w == 1
                seek(o, position(o)+n)
            elseif w == 2
                seekend(o)
                seek(o, position(o)+n)
            else
                pythrow(pyvalueerror("Unsupported whence: $w"))
            end
            pyint(position(o))
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("seek")))
    end
    t["__iter__"] = pymethod(o -> o)
    t["__next__"] = pymethod(o -> (x=o.readline(); pylen(x)==0 ? pythrow(pystopiteration()) : x))
end

### RawIO (as RawIOBase)

abstract type RawIO{V<:IO} <: PyJlSubclass{V} end

pyjlrawio(o::IO) = pyjl(o, RawIO{typeof(o)})

pyjl_abc(::Type{T}) where {T<:RawIO} = pyiomodule.RawIOBase

### BufferedIO (as BufferedIOBase)

abstract type BufferedIO{V<:IO} <: PyJlSubclass{V} end

pyjl_abc(::Type{T}) where {T<:BufferedIO} = pyiomodule.BufferedIOBase

pyjlbufferedio(o::IO) = pyjl(o, BufferedIO{typeof(o)})

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:BufferedIO, V}
    t["detach"] = pymethod(o -> pythrow(pyiounsupportedoperation("detach")))
    t["readinto"] = if hasmethod(readbytes!, Tuple{V, Vector{UInt8}})
        pymethod((_o, _b) -> begin
            b = PyBuffer(_b, C.PyBUF_WRITABLE)
            o = pyjlgetvalue(_o, V)
            n = readbytes!(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
            pyint(n)
        end)
    else
        pymethod((o, b) -> pythrow(pyiounsupportedoperation("readinto")))
    end
    t["read"] = if hasmethod(read, Tuple{V}) && hasmethod(read, Tuple{V, Int})
        pymethod((_o, size=pynone) -> begin
            n = pyconvert(Union{Int,Nothing}, size)
            o = pyjlgetvalue(_o, V)
            pybytes(convert(Vector{UInt8}, (n===nothing || n < 0) ? read(o) : read(o, n)))
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("read")))
    end
    t["write"] = if hasmethod(write, Tuple{V, Vector{UInt8}})
        pymethod((_o, _b) -> begin
            b = PyBuffer(_b, C.PyBUF_SIMPLE)
            o = pyjlgetvalue(_o, V)
            n = write(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
            pyint(n)
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("write")))
    end
    t["readline"] = if hasmethod(read, Tuple{V, Type{UInt8}})
        pymethod((_o, size=pynone) -> begin
            n = pyconvert(Union{Int,Nothing}, size)
            o = pyjlgetvalue(_o, V)
            data = UInt8[]
            while !eof(o) && (n===nothing || n < 0 || length(data) ≤ n)
                c = read(o, UInt8)
                push!(data, c)
                c == 0x0A && break
            end
            pybytes(data)
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("readline")))
    end
end

### TextIO (as TextIOBase)

abstract type TextIO{V<:IO} <: PyJlSubclass{V} end

pyjl_abc(::Type{T}) where {T<:TextIO} = pyiomodule.TextIOBase

pyjltextio(o::IO) = pyjl(o, TextIO{typeof(o)})

function pyjl_addattrs(t, ::Type{T}, ::Type{V}) where {T<:TextIO, V}
    t["encoding"] = pyproperty(o -> pystr("utf-8"))
    t["errors"] = pyproperty(o -> pystr("strict"))
    t["newlines"] = pyproperty(o -> pynone)
    t["detach"] = pymethod(o -> pythrow(pyiounsupportedoperation("detach")))
    t["read"] = if hasmethod(read, Tuple{V, Type{Char}})
        pymethod((_o, size=pynone) -> begin
            n = pytryconvert(Union{Nothing,Int}, size)
            n === PyConvertFail() && pythrow(pytypeerror("'size' must be 'int' or 'None'"))
            o = pyjlgetvalue(_o, V)
            b = IOBuffer()
            i = 0
            while !eof(o) && (n===nothing || n < 0 || i < n)
                i += 1
                write(b, read(o, Char))
            end
            seekstart(b)
            pystr(read(b, String))
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("read")))
    end
    t["readline"] = if hasmethod(read, Tuple{V, Type{Char}})
        pymethod((_o, size=pynone) -> begin
            n = pytryconvert(Union{Nothing,Int}, size)
            n === PyConvertFail() && pythrow(pytypeerror("'size' must be 'int' or 'None'"))
            o = pyjlgetvalue(_o, V)
            b = IOBuffer()
            i = 0
            while !eof(o) && (n === nothing || n < 0 || i < n)
                i += 1
                c = read(o, Char)
                if c == '\n'
                    write(b, '\n')
                    break
                elseif c == '\r'
                    write(b, '\n')
                    !eof(o) && peek(o, Char)=='\n' && read(o, Char)
                    break
                else
                    write(b, c)
                end
            end
            seekstart(b)
            pystr(read(b, String))
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("readline")))
    end
    t["write"] = if hasmethod(write, Tuple{V, String})
        pymethod((_o, _x) -> begin
            x = pystr_asjuliastring(_x)
            o = pyjlgetvalue(_o)
            n = 0
            linesep = pystr_asjuliastring(pyosmodule.linesep)
            i = firstindex(x)
            while true
                j = findnext('\n', x, i)
                if j === nothing
                    y = SubString(x, i, lastindex(x))
                    write(o, y)
                    n += length(y)
                    break
                else
                    y = SubString(x, i, prevind(x, j))
                    write(o, y)
                    write(o, linesep)
                    n += length(y) + 1
                    i = nextind(x, j)
                end
            end
            pyint(n)
        end)
    else
        pymethod((args...) -> pythrow(pyiounsupportedoperation("write")))
    end
end

### Buffer Protocol

isflagset(flags, mask) = (flags & mask) == mask

const PYJLBUFCACHE = Dict{Ptr{Cvoid}, Any}()

function pyjl_get_buffer_impl(o, buf, flags, ptr, elsz, len, ndim, fmt, sz, strds, mutable)
    b = UnsafePtr(buf)
    c = []

    # not influenced by flags: obj, buf, len, itemsize, ndim
    b.obj[] = C_NULL
    b.buf[] = ptr
    b.itemsize[] = elsz
    b.len[] = elsz * len
    b.ndim[] = ndim

    # readonly
    if isflagset(flags, C.PyBUF_WRITABLE)
        if mutable
            b.readonly[] = 1
        else
            pythrow(pybuffererror("not writable"))
        end
    else
        b.readonly[] = mutable ? 0 : 1
    end

    # format
    if isflagset(flags, C.PyBUF_FORMAT)
        b.format[] = cacheptr!(c, fmt)
    else
        b.format[] = C_NULL
    end

    # shape
    if isflagset(flags, C.PyBUF_ND)
        b.shape[] = cacheptr!(c, C.Py_ssize_t[sz...])
    else
        b.shape[] = C_NULL
    end

    # strides
    if isflagset(flags, C.PyBUF_STRIDES)
        b.strides[] = cacheptr!(c, C.Py_ssize_t[(strds .* elsz)...])
    else
        if size_to_cstrides(1, sz...) != strds
            pythrow(pybuffererror("not C contiguous and strides not requested"))
        end
        b.strides[] = C_NULL
    end

    # check contiguity
    if isflagset(flags, C.PyBUF_C_CONTIGUOUS)
        if size_to_cstrides(1, sz...) != strds
            pythrow(pybuffererror("not C contiguous"))
        end
    end
    if isflagset(flags, C.PyBUF_F_CONTIGUOUS)
        if size_to_fstrides(1, sz...) != strds
            pythrow(pybuffererror("not Fortran contiguous"))
        end
    end
    if isflagset(flags, C.PyBUF_ANY_CONTIGUOUS)
        if size_to_cstrides(1, sz...) != strds && size_to_fstrides(1, sz...) != strds
            pythrow(pybuffererror("not contiguous"))
        end
    end

    # suboffsets
    b.suboffsets[] = C_NULL

    # internal
    cptr = Base.pointer_from_objref(c)
    PYJLBUFCACHE[cptr] = c
    b.internal[] = cptr

    # obj
    b.obj[] = pyptr(pyincref!(o))
    Cint(0)
end

pyjl_get_buffer(o, buf, flags) = pyjl_get_buffer(o, buf, flags, pyjlgetvalue(o))
pyjl_get_buffer(o, buf, flags, x::AbstractArray) =
    pyjl_get_buffer_impl(o, buf, flags, pointer(x), Base.aligned_sizeof(eltype(x)), length(x), ndims(x), pybufferformat(eltype(x)), size(x), strides(x), ismutablearray(x))
pyjl_get_buffer(o, buf, flags, x::PyObjectArray) =
    pyjl_get_buffer_impl(o, buf, flags, pointer(x.ptrs), sizeof(CPyPtr), length(x), ndims(x), "O", size(x), strides(x.ptrs), true)
