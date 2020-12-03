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

function pyjlgetvalue(o::PyObject)
    pyisinstance(o, pyjlbasetype) || pythrow(pytypeerror("Expecting a Python 'julia.ValueBase'"))
    ptr = UnsafePtr(CPyJlPtr{Any}(pyptr(o))).value[!]
    ptr == C_NULL && pythrow(pyvalueerror("Value is NULL"))
    Base.unsafe_pointer_to_objref(ptr)
end
export pyjlgetvalue

### jlrawvalue

const PYJLRAWTYPES = Dict{Type,PyObject}()

pyjlrawtype(::Type{T}) where {T} = get!(PYJLRAWTYPES, T) do
    name = "julia.RawValue[$T]"
    base = T==Any ? pyjlbasetype : pyjlrawtype(T==DataType ? Type : supertype(T))
    t = pytypetype(name, (base,), pydict())
    t.__repr__ = pymethod(self -> "<jl $(repr(pyjlgetvalue(self)::T))>")
    t.__str__ = pymethod(self -> string(pyjlgetvalue(self)::T))
    t.__call__ = pymethod((self, args...; kwargs...) -> pyjlraw((pyjlgetvalue(self)::T)(map(pyjlgetvalue, args)...)))
    t._jl_getfield = pymethod((self, k) -> pyjlraw(getfield(pyjlgetvalue(self)::T, Symbol(string(k)))))
    t._jl_getproperty = pymethod((self, k) -> pyjlraw(getproperty(pyjlgetvalue(self)::T, Symbol(string(k)))))
    t._jl_setfield = pymethod((self, k, v) -> (setfield!(pyjlgetvalue(self)::T, Symbol(string(k)), pyjlgetvalue(v)); nothing))
    t._jl_setproperty = pymethod((self, k, v) -> (setproperty!(pyjlgetvalue(self)::T, Symbol(string(k)), pyjlgetvalue(v)); nothing))
    t.__getitem__ = pymethod((self, i) -> pyjlraw(getindex(pyjlgetvalue(self)::T, (pyistuple(i) ? [pyjlgetvalue(j) for j in i] : [pyjlgetvalue(i)])...)))
    t.__setitem__ = pymethod((self, i, v) -> pyjlraw(setindex!(pyjlgetvalue(self)::T, pyjlgetvalue(v), (pyistuple(i) ? [pyjlgetvalue(j) for j in i] : [pyjlgetvalue(i)])...)))
    t.__doc__ = """
        A Julia '$T' with basic Julia semantics.
        """
    t
end
export pyjlrawtype

pyjlraw(x::T, ::Type{T}) where {T} = pyjlnewvalue(x, pyjlrawtype(T))
pyjlraw(x) = pyjlraw(x, typeof(x))
export pyjlraw

### jlvalue

const PYJLTYPES = Dict{Type,PyObject}()

pyjltype(::Type{T}) where {T} = get!(PYJLTYPES, T) do
    # Pick a name
    name = "julia.Value[$T]"

    # Pick the sequence of base types
    bases = [T==Any ? pyjlbasetype : pyjltype(T==DataType ? Type : supertype(T))]

    # Make the type
    t = pytypetype(name, pytuple_fromiter(bases), pydict())

    # Add methods
    t.__repr__ = pymethod(self -> "<jl $(repr(pyjlgetvalue(self)::T))>")
    t.__str__ = pymethod(self -> string(pyjlgetvalue(self)::T))
    t.__doc__ = """
    A Julia '$T' with Python semantics.
    """
    if mighthavemethod(length, Tuple{T})
        t.__len__ = pymethod(self -> length(pyjlgetvalue(self)::T))
    end
    if mighthavemethod(hash, Tuple{T})
        t.__hash__ = pymethod(self -> hash(pyjlgetvalue(self)::T))
    end
    if mighthavemethod(in, Tuple{Any, T})
        t.__contains__ = pymethod((_o, _x) -> begin
            o = pyjlgetvalue(_o)::T
            x = pytryconvert_element(o, _x)
            x === PyConvertFail() ? false : in(x, o)
        end)
    end
    if mighthavemethod(reverse, Tuple{T})
        t.__reversed__ = pymethod(self -> pyjl(reverse(pyjlgetvalue(self)::T)))
    end
    if mighthavemethod(iterate, Tuple{T})
        t.__iter__ = pymethod(self -> pyjl(Iterator(pyjlgetvalue(self)::T, nothing)))
    end
    t.__dir__ = pymethod(self -> begin
        d = pyobjecttype.__dir__(self)
        d.extend(pylist([string(k) for k in propertynames(pyjlgetvalue(self)::T)]))
        d
    end)
    t.__call__ = pymethod((self, args...; kwargs...) -> pyobject((pyjlgetvalue(self)::T)(args...; kwargs...)))
    if mighthavemethod(getindex, Tuple{T, Any})
        t.__getitem__ = pymethod((_o, _k) -> begin
            o = pyjlgetvalue(_o)::T
            k = pytryconvert_indices(o, _k)
            if k === PyConvertFail()
                pythrow(pytypeerror("invalid index"))
            end
            pyobject(o[k...])
        end)
    end
    if mighthavemethod(setindex!, Tuple{T, Any, Any})
        t.__setitem__ = pymethod((_o, _k, _v) -> begin
            o = pyjlgetvalue(_o)::T
            k = pytryconvert_indices(o, _k)
            k === PyConvertFail && pythrow(pytypeerror("invalid index"))
            v = pytryconvert_value(o, _v, k...)
            v === PyConvertFail && pythrow(pytypeerror("invalid value"))
            o[k...] = v
            pynone
        end)
    end
    if T <: Iterator
        t.__iter__ = pymethod(identity)
        t.__next__ = pymethod(_o -> begin
            o = pyjlgetvalue(_o)::T
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
    t
end
export pyjltype

pyjl(x::T, ::Type{T}) where {T} = pyjlnewvalue(x, pyjltype(T))
pyjl(x) = pyjl(x, typeof(x))
export pyjl

mutable struct Iterator{T}
    val :: T
    st :: Union{Nothing, Some}
end

# pyjl_attrname_py2jl(x::AbstractString) =
#     replace(x, r"_[_b]" => s -> (s[2]=='b' ? '!' : '_'))

# pyjl_attrname_jl2py(x::AbstractString) =
#     replace(replace(x, r"_(?=[_b])" => "__"), '!'=>"_b")

# cpyjlattr(::Val{:__getattr__}, ::Type{T}, ::Type{V}) where {T, V} =
#     @cfunction (_o, _k) -> cpycatch() do
#         # first do the generic lookup
#         _x = C.PyObject_GenericGetAttr(_o, _k)
#         (_x == C_NULL && pyerroccurred(pyattributeerror)) || return _x
#         errstate = pyerrfetch()
#         # then see if there is a corresponding julia property
#         o = cpyjlvalue(_o)
#         k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pyborrowedobject(_k))))
#         if hasproperty(o, k)
#             return cpyreturn(CPyPtr, pyobject(getproperty(o, k)))
#         end
#         # no such attribute
#         pyerrrestore(errstate...)
#         return _x
#     end CPyPtr (CPyJlPtr{V}, CPyPtr)

# cpyjlattr(::Val{:__setattr__}, ::Type{T}, ::Type{V}) where {T,V} =
#     @cfunction (_o, _k, _v) -> cpycatch(Cint) do
#         # first do the generic version
#         err = C.PyObject_GenericSetAttr(_o, _k, _v)
#         (err == -1 && pyerroccurred(pyattributeerror)) || return err
#         errstate = pyerrfetch()
#         # now see if there is a corresponding julia property
#         _v == C_NULL && error("deletion not supported")
#         o = cpyjlvalue(_o)
#         k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pyborrowedobject(_k))))
#         if hasproperty(o, k)
#             v = pyconvert(Any, pyborrowedobject(_v)) # can we do better than Any?
#             setproperty!(o, k, v)
#             return Cint(0)
#         end
#         # no such attribute
#         pyerrrestore(errstate...)
#         return err
#     end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)

# #### ARRAY AS BUFFER AND ARRAY

# pyjlisbufferabletype(::Type{T}) where {T} =
#     T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Bool, Ptr{Cvoid})
# pyjlisbufferabletype(::Type{T}) where {T<:Tuple} =
#     isconcretetype(T) && Base.allocatedinline(T) && all(pyjlisbufferabletype, fieldtypes(T))
# pyjlisbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
#     pyjlisbufferabletype(T)

# isflagset(flags, mask) = (flags & mask) == mask

# const PYJLBUFCACHE = Dict{Ptr{Cvoid}, Any}()

# function cpyjl_getbuffer_impl(_o, _b, flags, ptr, elsz, len, ndim, fmt, sz, strds, mutable)
#     b = UnsafePtr(_b)
#     c = []

#     # not influenced by flags: obj, buf, len, itemsize, ndim
#     b.obj[] = C_NULL
#     b.buf[] = ptr
#     b.itemsize[] = elsz
#     b.len[] = elsz * len
#     b.ndim[] = ndim

#     # readonly
#     if isflagset(flags, C.PyBUF_WRITABLE)
#         if mutable
#             b.readonly[] = 1
#         else
#             pyerrset(pybuffererror, "not writable")
#             return -1
#         end
#     else
#         b.readonly[] = mutable ? 0 : 1
#     end

#     # format
#     if isflagset(flags, C.PyBUF_FORMAT)
#         b.format[] = cacheptr!(c, fmt)
#     else
#         b.format[] = C_NULL
#     end

#     # shape
#     if isflagset(flags, C.PyBUF_ND)
#         b.shape[] = cacheptr!(c, C.Py_ssize_t[sz...])
#     else
#         b.shape[] = C_NULL
#     end

#     # strides
#     if isflagset(flags, C.PyBUF_STRIDES)
#         b.strides[] = cacheptr!(c, C.Py_ssize_t[(strds .* elsz)...])
#     else
#         if size_to_cstrides(1, sz...) != strds
#             pyerrset(pybuffererror, "not C contiguous and strides not requested")
#             return -1
#         end
#         b.strides[] = C_NULL
#     end

#     # check contiguity
#     if isflagset(flags, C.PyBUF_C_CONTIGUOUS)
#         if size_to_cstrides(1, sz...) != strds
#             pyerrset(pybuffererror, "not C contiguous")
#             return -1
#         end
#     end
#     if isflagset(flags, C.PyBUF_F_CONTIGUOUS)
#         if size_to_fstrides(1, sz...) != strds
#             pyerrset(pybuffererror, "not Fortran contiguous")
#             return -1
#         end
#     end
#     if isflagset(flags, C.PyBUF_ANY_CONTIGUOUS)
#         if size_to_cstrides(1, sz...) != strds && size_to_fstrides(1, sz...) != strds
#             pyerrset(pybuffererror, "not contiguous")
#             return -1
#         end
#     end

#     # suboffsets
#     b.suboffsets[] = C_NULL

#     # internal
#     cptr = Base.pointer_from_objref(c)
#     PYJLBUFCACHE[cptr] = c
#     b.internal[] = cptr

#     # obj
#     b.obj[] = _o
#     C.Py_IncRef(_o)
#     return 0
# end

# function cpyjl_releasebuffer_impl(_o, _b)
#     b = UnsafePtr(_b)
#     delete!(PYJLBUFCACHE, b.internal[])
#     nothing
# end

# cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
#     if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
#         @cfunction (_o, _b, flags) -> cpycatch(Cint) do
#             o = cpyjlvalue(_o)
#             cpyjl_getbuffer_impl(_o, _b, flags, pointer(o), Base.aligned_sizeof(eltype(o)), length(o), ndims(o), pybufferformat(eltype(o)), size(o), strides(o), ismutablearray(o))
#         end Cint (CPyJlPtr{V}, Ptr{C.Py_buffer}, Cint)
#     end

# cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
#     if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
#         @cfunction (_o, _b) -> cpycatch(Cvoid) do
#             cpyjl_releasebuffer_impl(_o, _b)
#         end Cvoid (CPyJlPtr{V}, Ptr{C.Py_buffer})
#     end

# pyjlisarrayabletype(::Type{T}) where {T} =
#     T in (UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Bool, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64})

# islittleendian() = Base.ENDIAN_BOM == 0x04030201 ? true : Base.ENDIAN_BOM == 0x01020304 ? false : error()

# function pytypestrformat(::Type{T}) where {T}
#     c = islittleendian() ? '<' : '>'
#     T ==    Int8 ? ("$(c)i1", pynone) :
#     T ==   UInt8 ? ("$(c)u1", pynone) :
#     T ==   Int16 ? ("$(c)i2", pynone) :
#     T ==  UInt16 ? ("$(c)u2", pynone) :
#     T ==   Int32 ? ("$(c)i4", pynone) :
#     T ==  UInt32 ? ("$(c)u4", pynone) :
#     T ==   Int64 ? ("$(c)i8", pynone) :
#     T ==  UInt64 ? ("$(c)u8", pynone) :
#     T == Float16 ? ("$(c)f2", pynone) :
#     T == Float32 ? ("$(c)f4", pynone) :
#     T == Float64 ? ("$(c)f8", pynone) :
#     error("not implemented")
# end

# cpyjlattr(::Val{:__array_interface__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
#     if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisarrayabletype(T)
#         :property => Dict(
#             :get => @cfunction (_o, _) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 typestr, descr = pytypestrformat(eltype(o))
#                 pydict(
#                     shape = size(o),
#                     typestr = typestr,
#                     descr = descr,
#                     data = (UInt(pointer(o)), !ismutablearray(o)),
#                     strides = strides(o) .* Base.aligned_sizeof(eltype(o)),
#                     version = 3,
#                 )
#             end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#         )
#     end

# cpyjlattr(::Val{:__array__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth => @cfunction (_o, _) -> cpycatch() do
#             if C.PyObject_HasAttrString(_o, "__array_interface__") != 0
#                 pynumpy.asarray(pyborrowedobject(_o))
#             else
#                 pynumpy.array(PyObjectArray(cpyjlvalue(_o)))
#             end
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# #### PYOBJECTARRAY AS BUFFER AND ARRAY

# cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
#     @cfunction (_o, _b, flags) -> cpycatch(Cint) do
#         o = cpyjlvalue(_o)
#         cpyjl_getbuffer_impl(_o, _b, flags, pointer(o.ptrs), sizeof(CPyPtr), length(o), ndims(o), "O", size(o), strides(o.ptrs), true)
#     end Cint (CPyJlPtr{V}, Ptr{C.Py_buffer}, Cint)

# cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
#     @cfunction (_o, _b) -> cpycatch(Cvoid) do
#         cpyjl_releasebuffer_impl(_o, _b)
#     end Cvoid (CPyJlPtr{V}, Ptr{C.Py_buffer})

# cpyjlattr(::Val{:__array_interface__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
#     :property => Dict(
#         :get => @cfunction (_o, _) -> cpycatch() do
#             o = cpyjlvalue(_o)
#             pydict(
#                 shape = size(o),
#                 typestr = "O",
#                 data = (UInt(pointer(o.ptrs)), false),
#                 strides = strides(o.ptrs) .* sizeof(CPyPtr),
#                 version = 3,
#             )
#         end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#     )

# #### VECTOR, TUPLE, NAMED TUPLE as SEQUENCE

# const SequenceLike = Union{AbstractVector, Tuple, NamedTuple}

# pyjlabc(::Type{T}) where {T<:SequenceLike} = pycollectionsabcmodule.Sequence

# cpyjlattr(::Val{:__getitem__}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
#     if mighthavemethod(getindex, Tuple{T, Any})
#         @cfunction (_o, _k) -> cpycatch(CPyPtr) do
#             o = cpyjlvalue(_o)
#             k = pyborrowedobject(_k)
#             i = pytryconvert(Union{Int,Tuple{Vararg{Int}}}, k)
#             if i===PyConvertFail()
#                 pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
#                 return CPyPtr(0)
#             end
#             cpyreturn(CPyPtr, pyobject(o[(i.+1)...]))
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     end

# cpyjlattr(::Val{:__setitem__}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
#     if mighthavemethod(setindex!, Tuple{T, Any, Any})
#         @cfunction (_o, _k, _v) -> cpycatch(Cint) do
#             _v == C_NULL && error("deletion not implemented")
#             o = cpyjlvalue(_o)
#             k = pyborrowedobject(_k)
#             i = pytryconvert(Union{Int,Tuple{Vararg{Int}}}, k)
#             if i===PyConvertFail()
#                 pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
#                 return -1
#             end
#             v = pytryconvert_value(o, pyborrowedobject(_v), (i.+1)...)
#             o[(i.+1)...] = v
#             0
#         end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)
#     end

# cpyjlattr(::Val{:index}, ::Type{T}, ::Type{V}) where {T<:Union{SequenceLike}, V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth => @cfunction (_o, _v) -> cpycatch() do
#             o = cpyjlvalue(_o)
#             v = pytryconvert_value(o, pyborrowedobject(_v))
#             i = v===PyConvertFail() ? nothing : findfirst(==(v), o)
#             if i === nothing
#                 pyerrset(pyvalueerror, "no such value")
#             else
#                 pyint(i - firstindex(o))
#             end
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:count}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth => @cfunction (_o, _v) -> cpycatch() do
#             o = cpyjlvalue(_o)
#             v = pytryconvert_value(o, pyborrowedobject(_v))
#             n = v===PyConvertFail() ? 0 : count(==(v), o)
#             pyint(n)
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:append}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(push!, Tuple{T,Any})
#         :method => Dict(
#             :flags => C.Py_METH_O,
#             :meth => @cfunction (_o, _v) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 v = pytryconvert_value(o, pyborrowedobject(_v))
#                 if v === PyConvertFail()
#                     pyerrset(pytypeerror, "invalid element type")
#                     CPyPtr(0)
#                 else
#                     push!(o, v)
#                     cpyreturn(CPyPtr, pynone)
#                 end
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:__iconcat__}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     @cfunction (_o, _v) -> cpycatch() do
#         o = cpyjlvalue(_o)
#         for _x in pyborrowedobject(_v)
#             x = pytryconvert_value(o, _x)
#             if x === PyConvertFail()
#                 pyerrset(pytypeerror, "invalid element type")
#                 return CPyPtr(0)
#             else
#                 push!(o, x)
#             end
#         end
#         cpyreturn(CPyPtr, pynone)
#     end CPyPtr (CPyJlPtr{V}, CPyPtr)

# cpyjlattr(::Val{:extend}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(push!, Tuple{T,Any})
#         :method => Dict(
#             :flags => C.Py_METH_O,
#             :meth => @cfunction (_o, _v) -> cpycatch() do
#                 pyborrowedobject(_o).__iadd__(pyborrowedobject(_v))
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:insert}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(insert!, Tuple{T,Int,Any})
#         :method => Dict(
#             :flags => C.Py_METH_VARARGS,
#             :meth => @cfunction (_o, _args) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 i, _v = @pyconvert_args (i::Int, x) pyborrowedobject(_args)
#                 v = pytryconvert_value(o, _v)
#                 if v === PyConvertFail()
#                     pyerrset(pytypeerror, "invalid element type")
#                     CPyPtr(0)
#                 else
#                     insert!(o, i + firstindex(o), v)
#                     cpyreturn(CPyPtr, pynone)
#                 end
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:pop}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(popat!, Tuple{T,Int})
#         :method => Dict(
#             :flags => C.Py_METH_VARARGS,
#             :meth => @cfunction (_o, _args) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 i, = @pyconvert_args (i::Union{Int,Nothing}=nothing,) pyborrowedobject(_args)
#                 pyobject(popat!(o, i===nothing ? lastindex(o) : i + firstindex(o)))
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:remove}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(popat!, Tuple{T,Int})
#         :method => Dict(
#             :flags => C.Py_METH_O,
#             :meth => @cfunction (_o, _v) -> cpycatch() do
#                 o = pyborrowedobject(_o)
#                 v = pyborrowedobject(_v)
#                 o.pop(o.index(v))
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:clear}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(empty!, Tuple{T})
#         :method => Dict(
#             :flags => C.Py_METH_NOARGS,
#             :meth => @cfunction (_o, _) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 empty!(o)
#                 pynone
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:sort}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(sort!, Tuple{T})
#         :method => Dict(
#             :flags => C.Py_METH_VARARGS | C.Py_METH_KEYWORDS,
#             :meth => @cfunction (_o, _args, _kwargs) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 key, rev = @pyconvert_args (key=pynone, rev::Bool=false) pyborrowedobject(_args) (_kwargs == C_NULL ? pydict() : pyborrowedobject(_kwargs))
#                 sort!(o; by = pyisnone(key) ? identity : key, rev = rev)
#                 pynone
#             end CPyPtr (CPyJlPtr{V}, CPyPtr, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:reverse}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(reverse!, Tuple{T})
#         :method => Dict(
#             :flags => C.Py_METH_NOARGS,
#             :meth => @cfunction (_o, _) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 reverse!(o)
#                 pynone
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# cpyjlattr(::Val{:copy}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
#     if mighthavemethod(copy, Tuple{T})
#         :method => Dict(
#             :flags => C.Py_METH_NOARGS,
#             :meth => @cfunction (_o, _) -> cpycatch() do
#                 o = cpyjlvalue(_o)
#                 pyjl(copy(o))
#             end CPyPtr (CPyJlPtr{V}, CPyPtr)
#         )
#     end

# #### SET AS SET

# pyjlabc(::Type{T}) where {T<:AbstractSet} = pycollectionsabcmodule.Set

# #### DICT AS MAPPING

# pyjlabc(::Type{T}) where {T<:AbstractDict} = pycollectionsabcmodule.Mapping

# #### NUMBER AS NUMBER

# pyjlabc(::Type{T}) where {T<:Number} = pynumbersmodule.Number
# pyjlabc(::Type{T}) where {T<:Complex} = pynumbersmodule.Complex
# pyjlabc(::Type{T}) where {T<:Real} = pynumbersmodule.Real
# pyjlabc(::Type{T}) where {T<:Rational} = pynumbersmodule.Rational
# pyjlabc(::Type{T}) where {T<:Integer} = pynumbersmodule.Integral

# #### IO AS IO

# abstract type RawIO{V} <: SubClass{V} end
# abstract type BufferedIO{V} <: SubClass{V} end
# abstract type TextIO{V} <: SubClass{V} end

# pyjlabc(::Type{T}) where {T<:IO} = pyiomodule.IOBase
# pyjlabc(::Type{T}) where {T<:RawIO} = pyiomodule.RawIOBase
# pyjlabc(::Type{T}) where {T<:BufferedIO} = pyiomodule.BufferedIOBase
# pyjlabc(::Type{T}) where {T<:TextIO} = pyiomodule.TextIOBase

# pyjlrawio(o::IO) = pyjl(o, RawIO{typeof(o)})
# pyjlbufferedio(o::IO) = pyjl(o, BufferedIO{typeof(o)})
# pyjltextio(o::IO) = pyjl(o, TextIO{typeof(o)})

# # RawIO
# # TODO:
# # readline(size=-1)
# # readlines(hint=-1)

# cpyjlattr(::Val{:close}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth => @cfunction (_o, _) -> cpycatch() do
#             close(cpyjlvalue(_o))
#             pynone
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:closed}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :property => Dict(
#         :get => @cfunction (_o, _) -> cpycatch() do
#             pybool(!isopen(cpyjlvalue(_o)))
#         end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#     )

# cpyjlattr(::Val{:fileno}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(fd, Tuple{T})
#                 @cfunction (_o, _) -> cpycatch() do
#                     pyint(fd(cpyjlvalue(_o)))
#                 end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "fileno"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:flush}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(flush, Tuple{T})
#                 @cfunction (_o, _) -> cpycatch() do
#                     flush(cpyjlvalue(_o))
#                     pynone
#                 end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "flush"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:isatty}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if T <: Base.TTY
#                 @cfunction (_o, _) -> cpyreturn(CPyPtr, pytrue) CPyPtr (CPyPtr, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> cpyreturn(CPyPtr, pyfalse) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:readable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(isreadable, Tuple{T})
#                 @cfunction (_o, _) -> cpycatch() do
#                     pybool(isreadable(cpyjlvalue(_o)))
#                 end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readable"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:writable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(iswritable, Tuple{T})
#                 @cfunction (_o, _) -> cpycatch() do
#                     pybool(iswritable(cpyjlvalue(_o)))
#                 end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "writable"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:tell}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(position, Tuple{T})
#                 @cfunction (_o, _) -> cpycatch() do
#                     pyint(position(cpyjlvalue(_o)))
#                 end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "tell"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:writelines}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth => @cfunction (_o, _lines) -> cpycatch() do
#             wr = pyborrowedobject(_o).write
#             for line in pyborrowedobject(_lines)
#                 wr(line)
#             end
#             pyreturn(CPyPtr, pynone)
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:seekable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth =>
#             if mighthavemethod(position, Tuple{T}) && mighthavemethod(seek, Tuple{T,Int}) && mighthavemethod(truncate, Tuple{T,Int})
#                 @cfunction (_o, _) -> cpyreturn(CPyPtr, pytrue) CPyPtr (CPyPtr, Ptr{Cvoid})
#             else
#                 @cfunction (_o, _) -> cpyreturn(CPyPtr, pyfalse) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:truncate}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth =>
#             if mighthavemethod(truncate, Tuple{T, Int}) && mighthavemethod(position, Tuple{T})
#                 @cfunction (_o, _args) -> cpycatch() do
#                     n, = @pyconvert_args (size::Union{Int,Nothing}=nothing,) pyborrowedobject(_args)
#                     n === nothing && (n = position(o))
#                     truncate(o, n)
#                     pyint(n)
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "truncate"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:seek}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth =>
#             if mighthavemethod(seek, Tuple{V, Int}) && mighthavemethod(position, Tuple{V})
#                 @cfunction (_o, _args) -> cpycatch() do
#                     n, w = @pyconvert_args (offset::Int, whence::Int=0) pyborrowedobject(_args)
#                     o = cpyjlvalue(_o)
#                     if w == 0
#                         seek(o, n)
#                     elseif w==1
#                         seek(o, position(o)+n)
#                     elseif w==2
#                         seekend(o)
#                         seek(o, position(o)+n)
#                     else
#                         pythrow(pyvalueerror("invalid whence: $w"))
#                     end
#                     pyint(position(o))
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "seek"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
#             end
#     )

# cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     @cfunction o -> (C.Py_IncRef(o); CPyPtr(o)) CPyPtr (CPyJlPtr{V},)

# cpyjlattr(::Val{:__next__}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
#     @cfunction _o -> cpycatch() do
#         line = pyborrowedobject(_o).readline()
#         pylen(line) == 0 ? C_NULL : cpyreturn(CPyPtr, line)
#     end CPyPtr (CPyJlPtr{V},)

# # BufferedIO
# # TODO:
# # read1(size=-1)
# # readinto1(b)

# cpyjlattr(::Val{:detach}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth => @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "detach"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:readinto}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth =>
#             if mighthavemethod(readbytes!, Tuple{V, Vector{UInt8}})
#                 @cfunction (_o, _b) -> cpycatch() do
#                     b = PyBuffer(pyborrowedobject(_b), C.PyBUF_WRITABLE)
#                     o = cpyjlvalue(_o)
#                     n = readbytes!(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
#                     pyint(n)
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readinto"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#             end
#     )

# cpyjlattr(::Val{:read}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth =>
#             if mighthavemethod(read, Tuple{V}) && mighthavemethod(read, Tuple{V, Int})
#                 @cfunction (_o, _args) -> cpycatch() do
#                     n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pyborrowedobject(_args)
#                     o = cpyjlvalue(_o)
#                     pybytes(convert(Vector{UInt8}, (n===nothing || n < 0) ? read(o) : read(o, n)))
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "read"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#             end
#     )

# cpyjlattr(::Val{:write}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth =>
#             if mighthavemethod(write, Tuple{V, Vector{UInt8}})
#                 @cfunction (_o, _b) -> cpycatch() do
#                     b = PyBuffer(pyborrowedobject(_b), C.PyBUF_SIMPLE)
#                     o = cpyjlvalue(_o)
#                     n = write(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
#                     pyint(n)
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "write"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#             end
#     )

# cpyjlattr(::Val{:readline}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth =>
#             if mighthavemethod(read, Tuple{V, Type{UInt8}})
#                 @cfunction (_o, _args) -> cpycatch() do
#                     n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pyborrowedobject(_args)
#                     o = cpyjlvalue(_o)
#                     data = UInt8[]
#                     while !eof(o) && (n===nothing || n < 0 || length(data) ≤ n)
#                         c = read(o, UInt8)
#                         push!(data, c)
#                         c == 0x0A && break
#                     end
#                     pybytes(data)
#                 end CPyPtr (CPyJlPtr{V}, CPyPtr)
#             else
#                 @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readline"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#             end
#     )

# # TextIO

# cpyjlattr(::Val{:encoding}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :property => Dict(
#         :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pystr("utf-8")) CPyPtr (CPyPtr, Ptr{Cvoid})
#     )

# cpyjlattr(::Val{:errors}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :property => Dict(
#         :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pystr("strict")) CPyPtr (CPyPtr, Ptr{Cvoid})
#     )

# cpyjlattr(::Val{:newlines}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :property => Dict(
#         :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pynone) CPyPtr (CPyPtr, Ptr{Cvoid})
#     )

# cpyjlattr(::Val{:detach}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_NOARGS,
#         :meth => @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "detach"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:read}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth => @cfunction (_o, _args) -> cpycatch() do
#             n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pyborrowedobject(_args)
#             o = cpyjlvalue(_o)
#             b = IOBuffer()
#             i = 0
#             while !eof(o) && (n===nothing || n < 0 || i < n)
#                 i += 1
#                 write(b, read(o, Char))
#             end
#             seekstart(b)
#             pystr(read(b, String))
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:readline}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_VARARGS,
#         :meth => @cfunction (_o, _args) -> cpycatch() do
#             n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pyborrowedobject(_args)
#             o = cpyjlvalue(_o)
#             b = IOBuffer()
#             i = 0
#             while !eof(o) && (n===nothing || n < 0 || i < n)
#                 i += 1
#                 c = read(o, Char)
#                 if c == '\n'
#                     write(b, '\n')
#                     break
#                 elseif c == '\r'
#                     write(b, '\n')
#                     !eof(o) && peek(o, Char)=='\n' && read(o, Char)
#                     break
#                 else
#                     write(b, c)
#                 end
#             end
#             seekstart(b)
#             pystr(read(b, String))
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )

# cpyjlattr(::Val{:write}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
#     :method => Dict(
#         :flags => C.Py_METH_O,
#         :meth => @cfunction (_o, _x) -> cpycatch() do
#             x = pystr_asjuliastring(pyborrowedobject(_x))
#             o = cpyjlvalue(_o)
#             n = 0
#             linesep = pystr_asjuliastring(pyosmodule.linesep)
#             for c in x
#                 if c == '\n'
#                     write(o, linesep)
#                 else
#                     write(o, c)
#                 end
#                 n += 1
#             end
#             pyint(n)
#         end CPyPtr (CPyJlPtr{V}, CPyPtr)
#     )
