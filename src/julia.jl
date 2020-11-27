mighthavemethod(f, t) = !isempty(methods(f, t))

abstract type SubClass{V} end

pyjl_supertype(::Type{T}) where {T} = supertype(T)
pyjl_supertype(::Type{Any}) = nothing
pyjl_supertype(::Type{DataType}) = Type
pyjl_supertype(::Type{T}) where {V, T<:SubClass{V}} =
    if T == SubClass{V}
        error()
    elseif supertype(T) == SubClass{V}
        V
    else
        supertype(T)
    end

pyjl_valuetype(::Type{T}) where {T} = T
pyjl_valuetype(::Type{T}) where {V, T<:SubClass{V}} = V

@kwdef struct CPyJuliaTypeObject{T}
    ob_base :: C.PyTypeObject = C.PyTypeObject()
end

@kwdef struct CPyJuliaObject{T}
    ob_base :: C.PyObject = C.PyObject()
    value :: Ptr{Cvoid} = C_NULL
    weaklist :: CPyPtr = C_NULL
end
const CPyJlPtr{T} = Ptr{CPyJuliaObject{T}}

const PYJLTYPES = Dict{Type, PyObject}()
const PYJLGCCACHE = Dict{CPyPtr, Any}()
const PYJLBUFCACHE = Dict{Ptr{Cvoid}, Any}()

const pyjlexception = PyLazyObject() do
    # make the type
    c = []
    t = cpynewtype!(c; name="julia.JuliaException", base=pyexception, basicsize=0)
    # put into a 0-dim array and take a pointer
    ta = fill(t)
    ptr = pointer(ta)
    # ready the type
    check(C.PyType_Ready(ptr))
    # success
    PYJLGCCACHE[CPyPtr(ptr)] = push!(c, ta)
    pynewobject(ptr, true)
end
export pyjlexception

function pyjltype(::Type{T}) where {T}
    # see if we already made this type
    r = get(PYJLTYPES, T, nothing)
    r === nothing || return r

    # get attributes
    V = pyjl_valuetype(T)
    S = T
    attrnames = Set{Symbol}()
    special = Dict{Symbol, Ptr{Cvoid}}()
    methods = []
    getset = []
    while true
        V <: pyjl_valuetype(S) || error("value type of $T is $V, but value type of supertype $S is $(pyjl_valuetype(S))")
        for k in pyjlattrnames(S)
            if k ∉ attrnames
                a = cpyjlattr(Val(k), S, V)
                a === nothing && continue
                push!(attrnames, k)
                if a isa Ptr{Cvoid}
                    special[k] = a
                elseif a[1] == :method
                    push!(methods, Dict(pairs(a[2])..., :name=>string(k)))
                elseif a[1] == :property
                    push!(getset, Dict(pairs(a[2])..., :name=>string(k)))
                else
                    error("invalid attr $k: $a")
                end
            end
        end
        S = pyjl_supertype(S)
        S === nothing && break
    end

    # number methods
    nb_opts = Dict(
        :bool => pop!(special, :__bool__, C_NULL),
        :int => pop!(special, :__int__, C_NULL),
        :float => pop!(special, :__float__, C_NULL),
        :index => pop!(special, :__index__, C_NULL),
        :negative => pop!(special, :__neg__, C_NULL),
        :positive => pop!(special, :__pos__, C_NULL),
        :absolute => pop!(special, :__abs__, C_NULL),
        :invert => pop!(special, :__invert__, C_NULL),
    )
    filter!(x -> x.second != C_NULL, nb_opts)
    isempty(nb_opts) && (nb_opts = C_NULL)

    # mapping methods
    mp_opts = Dict(
        :length => pop!(special, :__len__, C_NULL),
        :subscript => pop!(special, :__getitem__, C_NULL),
        :ass_subscript => pop!(special, :__setitem__, C_NULL),
    )
    filter!(x -> x.second != C_NULL, mp_opts)
    isempty(mp_opts) && (mp_opts = C_NULL)

    # sequence methods
    sq_opts = Dict(
        :length => pop!(special, :__len__, C_NULL),
        :item => pop!(special, :__getitem_int__, C_NULL),
        :ass_item => pop!(special, :__setitem_int__, C_NULL),
        :contains => pop!(special, :__contains__, C_NULL),
        :concat => pop!(special, :__concat__, C_NULL),
        :inplace_concat => pop!(special, :__iconcat__, C_NULL),
        :repeat => pop!(special, :__repeat__, C_NULL),
        :inplace_repeat => pop!(special, :__irepeat__, C_NULL),
    )
    filter!(x -> x.second != C_NULL, sq_opts)
    isempty(sq_opts) && (sq_opts = C_NULL)

    # buffer methods
    buf_opts = Dict(
        :get => pop!(special, :__getbuffer__, C_NULL),
        :release => pop!(special, :__releasebuffer__, C_NULL),
    )
    filter!(x -> x.second != C_NULL, buf_opts)
    isempty(buf_opts) && (buf_opts = C_NULL)

    # type options
    opts = Dict{Symbol, Any}(
        :name => "julia.$T",
        :base => pyjl_supertype(T) === nothing ? pyobjecttype : pyjltype(pyjl_supertype(T)),
        :flags => C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG | (CONFIG.isstackless ? C.Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
        :basicsize => sizeof(CPyJuliaObject{T}),
        :dealloc => pop!(special, :__dealloc__, C_NULL),
        :hash => pop!(special, :__hash__, C_NULL),
        :repr => pop!(special, :__repr__, C_NULL),
        :str => pop!(special, :__str__, C_NULL),
        :iter => pop!(special, :__iter__, C_NULL),
        :iternext => pop!(special, :__next__, C_NULL),
        :getattr => pop!(special, :__getattr_str__, C_NULL),
        :setattr => pop!(special, :__setattr_str__, C_NULL),
        :getattro => pop!(special, :__getattr__, C_NULL),
        :setattro => pop!(special, :__setattr__, C_NULL),
        :call => pop!(special, :__call__, C_NULL),
        :as_number => nb_opts,
        :as_mapping => mp_opts,
        :as_sequence => sq_opts,
        :as_buffer => buf_opts,
        :methods => methods,
        :getset => getset,
    )

    # check we used all the special attributes
    isempty(special) || @warn "unused special attributes for julia type" T attrs=collect(keys(special))

    # make the type
    c = []
    t = CPyJuliaTypeObject{T}(ob_base=cpynewtype!(c; opts...))

    # put into a 0-dim array and take a pointer
    ta = fill(t)
    ptr = pointer(ta)

    # ready the type
    check(C.PyType_Ready(ptr))

    # cache
    PYJLGCCACHE[CPyPtr(ptr)] = push!(c, ta)
    r = PYJLTYPES[T] = pynewobject(ptr, true)

    # register ABC
    abc = pyjlabc(T)
    abc === nothing || abc.register(r)

    # done
    return r
end
export pyjltype

function pointer_from_obj(o::T) where {T}
    if T.mutable
        c = o
        p = Base.pointer_from_objref(o)
    else
        c = Ref{Any}(o)
        p = unsafe_load(Ptr{Ptr{Cvoid}}(Base.pointer_from_objref(c)))
    end
    p, c
end

function pyjl(x, ::Type{T}) where {T}
    x isa pyjl_valuetype(T) || error("x must be a `$(pyjl_valuetype(T))`")
    # get the type of the object
    t = pyjltype(T)
    # allocate an object
    o = check(C._PyObject_New(t))
    # set weakrefs and value
    p = UnsafePtr{CPyJuliaObject{T}}(pyptr(o))
    p.weaklist[] = C_NULL
    p.value[], PYJLGCCACHE[pyptr(o)] = pointer_from_obj(x)
    # done
    return o
end
pyjl(x) = pyjl(x, typeof(x))
export pyjl

pyisjulia(o::AbstractPyObject) = pytypecheck(o, pyjltype(Any))
export pyisjulia

function pyjlvalue(o::AbstractPyObject)
    if pyisjulia(o)
        GC.@preserve o cpyjlvalue(pyptr(o))
    else
        pythrow(pytypeerror("not a julia object"))
    end
end
export pyjlvalue

cpyjlvalue(p::Ptr{CPyJuliaObject{T}}) where {T} = Base.unsafe_pointer_to_objref(UnsafePtr(p).value[!]) :: pyjl_valuetype(T)
cpyjlvalue(p::Ptr) = cpyjlvalue(Ptr{CPyJuliaObject{Any}}(p))

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
                pyjl((err, bt))
            catch
                pynone
            end
            pyerrset(pyjlexception, val)
        end
        cpyerrval(T)
    end
end

cpyreturn(::Type{T}, x::T) where {T} = x
cpyreturn(::Type{CPyPtr}, x::AbstractPyObject) = CPyPtr(pyptr(pyincref!(x)))
cpyreturn(::Type{T}, x::Number) where {T<:Number} = convert(T, x)
cpyreturn(::Type{T}, x::Ptr) where {T<:Ptr} = T(x)

cpyerrval(::Type{Nothing}) = nothing
cpyerrval(::Type{T}) where {T<:Number} = zero(T)-one(T)
cpyerrval(::Type{T}) where {T<:Ptr} = T(C_NULL)

###################################################################################
# ABSTRACT BASE CLASSES

pyjlabc(::Type{T}) where {T} = nothing

###################################################################################
# ATTRIBUTE DEFINITIONS

function pyjlattrnames(::Type{T}) where {T}
    r = Symbol[]
    for m in methods(cpyjlattr, Tuple{Val, Type{T}, Type})
        sig = m.sig
        while sig isa UnionAll
            sig = sig.body
        end
        k = sig.parameters[2].parameters[1]
        push!(r, k)
    end
    r
end

cpyjlattr(::Val{:__dealloc__}, ::Type{T}, ::Type{V}) where {T, V} =
    @cfunction function (_o)
        UnsafePtr(_o).weaklist[!] == C_NULL || C.PyObject_ClearWeakRefs(_o)
        delete!(PYJLGCCACHE, CPyPtr(_o))
        nothing
    end Cvoid (CPyJlPtr{V},)

cpyjlattr(::Val{:__repr__}, ::Type{T}, ::Type{V}) where {T, V} =
    @cfunction _o -> cpycatch() do
        pystr(string("<jl ", repr(cpyjlvalue(_o)), ">"))
    end CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__str__}, ::Type{T}, ::Type{V}) where {T, V} =
    @cfunction _o -> cpycatch() do
        pystr(string(cpyjlvalue(_o)))
    end CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__len__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(length, Tuple{T})
        @cfunction _o -> cpycatch(C.Py_ssize_t) do
            length(cpyjlvalue(_o))
        end C.Py_ssize_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__hash__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(hash, Tuple{T})
        @cfunction _o -> cpycatch(C.Py_hash_t) do
            h = hash(cpyjlvalue(_o)) % C.Py_hash_t
            h == zero(C.Py_hash_t)-one(C.Py_hash_t) ? zero(C.Py_hash_t) : h
        end C.Py_hash_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__contains__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(in, Tuple{Any, T})
        @cfunction (_o, _v) -> cpycatch(Cint) do
            o = cpyjlvalue(_o)
            v = pytryconvert_element(o, pynewobject(_v, true))
            v === PyConvertFail() ? false : in(v, o)
        end Cint (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__reversed__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(reverse, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch(CPyPtr) do
                pyjl(reverse(cpyjlvalue(_o)))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:__getitem__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(getindex, Tuple{T, Any})
        @cfunction (_o, _k) -> cpycatch(CPyPtr) do
            o = cpyjlvalue(_o)
            k = pynewobject(_k, true)
            i = pytryconvert_indices(o, k)
            if i===PyConvertFail()
                pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
                return CPyPtr(0)
            end
            cpyreturn(CPyPtr, pyobject(o[i...]))
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__setitem__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(setindex!, Tuple{T, Any, Any})
        @cfunction (_o, _k, _v) -> cpycatch(Cint) do
            _v == C_NULL && error("deletion not implemented")
            o = cpyjlvalue(_o)
            k = pynewobject(_k, true)
            i = pytryconvert_indices(o, k)
            if i===PyConvertFail()
                pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
                return -1
            end
            v = pytryconvert_value(o, pynewobject(_v, true), i...)
            o[i...] = v
            0
        end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)
    end

mutable struct Iterator{T}
    val :: T
    st :: Union{Nothing, Some}
end

cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(iterate, Tuple{T})
        @cfunction _o -> cpycatch() do
            pyjl(Iterator(cpyjlvalue(_o), nothing))
        end CPyPtr (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T<:Iterator, V} =
    @cfunction _o->(cpyincref(_o); CPyPtr(_o)) CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__next__}, ::Type{T}, ::Type{V}) where {T<:Iterator, V} =
    @cfunction _o -> cpycatch() do
        o = cpyjlvalue(_o)
        if o.st === nothing
            it = iterate(o.val)
        else
            it = iterate(o.val, something(o.st))
        end
        if it === nothing
            cpyreturn(CPyPtr, C_NULL)
        else
            x, st = it
            o.st = Some(st)
            cpyreturn(CPyPtr, pyobject(x))
        end
    end  CPyPtr (CPyJlPtr{V},)

pyjl_attrname_py2jl(x::AbstractString) =
    replace(x, r"_[_b]" => s -> (s[2]=='b' ? '!' : '_'))

pyjl_attrname_jl2py(x::AbstractString) =
    replace(replace(x, r"_(?=[_b])" => "__"), '!'=>"_b")

cpyjlattr(::Val{:__getattr__}, ::Type{T}, ::Type{V}) where {T, V} =
    @cfunction (_o, _k) -> cpycatch() do
        # first do the generic lookup
        _x = C.PyObject_GenericGetAttr(_o, _k)
        (_x == C_NULL && pyerroccurred(pyattributeerror)) || return _x
        errstate = pyerrfetch()
        # then see if there is a corresponding julia property
        o = cpyjlvalue(_o)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pynewobject(_k, true))))
        if hasproperty(o, k)
            return cpyreturn(CPyPtr, pyobject(getproperty(o, k)))
        end
        # no such attribute
        pyerrrestore(errstate...)
        return _x
    end CPyPtr (CPyJlPtr{V}, CPyPtr)

cpyjlattr(::Val{:__setattr__}, ::Type{T}, ::Type{V}) where {T,V} =
    @cfunction (_o, _k, _v) -> cpycatch(Cint) do
        # first do the generic version
        err = C.PyObject_GenericSetAttr(_o, _k, _v)
        (err == -1 && pyerroccurred(pyattributeerror)) || return err
        errstate = pyerrfetch()
        # now see if there is a corresponding julia property
        _v == C_NULL && error("deletion not supported")
        o = cpyjlvalue(_o)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pynewobject(_k, true))))
        if hasproperty(o, k)
            v = pyconvert(Any, pynewobject(_v, true)) # can we do better than Any?
            setproperty!(o, k, v)
            return Cint(0)
        end
        # no such attribute
        pyerrrestore(errstate...)
        return err
    end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)

cpyjlattr(::Val{:__dir__}, ::Type{T}, ::Type{V}) where {T, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> cpycatch() do
            # default properties
            r = pyobjecttype.__dir__(pynewobject(_o, true))
            # add julia properties
            o = cpyjlvalue(_o)
            for k in propertynames(o)
                r.append(pystr(k))
            end
            r
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:__call__}, ::Type{T}, ::Type{V}) where {T,V} =
    @cfunction (_o, _args, _kwargs) -> cpycatch() do
        o = cpyjlvalue(_o)
        args = pynewobject(_args, true)
        kwargs = _kwargs==C_NULL ? Dict() : Dict(Symbol(pystr_asjuliastring(k)) => v for (k,v) in pynewobject(_kwargs, true))
        pyobject(o(args...; kwargs...))
    end CPyPtr (CPyJlPtr{V}, CPyPtr, CPyPtr)

#### ARRAY AS BUFFER AND ARRAY

pyjlisbufferabletype(::Type{T}) where {T} =
    T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Bool, Ptr{Cvoid})
pyjlisbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) && Base.allocatedinline(T) && all(pyjlisbufferabletype, fieldtypes(T))
pyjlisbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjlisbufferabletype(T)

isflagset(flags, mask) = (flags & mask) == mask

function cpyjl_getbuffer_impl(_o, _b, flags, ptr, elsz, len, ndim, fmt, sz, strds, mutable)
    b = UnsafePtr(_b)
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
            pyerrset(pybuffererror, "not writable")
            return -1
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
            pyerrset(pybuffererror, "not C contiguous and strides not requested")
            return -1
        end
        b.strides[] = C_NULL
    end

    # check contiguity
    if isflagset(flags, C.PyBUF_C_CONTIGUOUS)
        if size_to_cstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not C contiguous")
            return -1
        end
    end
    if isflagset(flags, C.PyBUF_F_CONTIGUOUS)
        if size_to_fstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not Fortran contiguous")
            return -1
        end
    end
    if isflagset(flags, C.PyBUF_ANY_CONTIGUOUS)
        if size_to_cstrides(1, sz...) != strds && size_to_fstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not contiguous")
            return -1
        end
    end

    # suboffsets
    b.suboffsets[] = C_NULL

    # internal
    cptr = Base.pointer_from_objref(c)
    PYJLBUFCACHE[cptr] = c
    b.internal[] = cptr

    # obj
    b.obj[] = _o
    C.Py_IncRef(_o)
    return 0
end

function cpyjl_releasebuffer_impl(_o, _b)
    b = UnsafePtr(_b)
    delete!(PYJLBUFCACHE, b.internal[])
    nothing
end

cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b, flags) -> cpycatch(Cint) do
            o = cpyjlvalue(_o)
            cpyjl_getbuffer_impl(_o, _b, flags, pointer(o), Base.aligned_sizeof(eltype(o)), length(o), ndims(o), pybufferformat(eltype(o)), size(o), strides(o), ismutablearray(o))
        end Cint (CPyJlPtr{V}, Ptr{C.Py_buffer}, Cint)
    end

cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b) -> cpycatch(Cvoid) do
            cpyjl_releasebuffer_impl(_o, _b)
        end Cvoid (CPyJlPtr{V}, Ptr{C.Py_buffer})
    end

pyjlisarrayabletype(::Type{T}) where {T} =
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

cpyjlattr(::Val{:__array_interface__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisarrayabletype(T)
        :property => Dict(
            :get => @cfunction (_o, _) -> cpycatch() do
                o = cpyjlvalue(_o)
                typestr, descr = pytypestrformat(eltype(o))
                pydict(
                    shape = size(o),
                    typestr = typestr,
                    descr = descr,
                    data = (UInt(pointer(o)), !ismutablearray(o)),
                    strides = strides(o) .* Base.aligned_sizeof(eltype(o)),
                    version = 3,
                )
            end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
        )
    end

cpyjlattr(::Val{:__array__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> cpycatch() do
            if C.PyObject_HasAttrString(_o, "__array_interface__") != 0
                pynumpy.asarray(pynewobject(_o, true))
            else
                pynumpy.array(PyObjectArray(cpyjlvalue(_o)))
            end
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

#### PYOBJECTARRAY AS BUFFER AND ARRAY

cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    @cfunction (_o, _b, flags) -> cpycatch(Cint) do
        o = cpyjlvalue(_o)
        cpyjl_getbuffer_impl(_o, _b, flags, pointer(o.ptrs), sizeof(CPyPtr), length(o), ndims(o), "O", size(o), strides(o.ptrs), true)
    end Cint (CPyJlPtr{V}, Ptr{C.Py_buffer}, Cint)

cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    @cfunction (_o, _b) -> cpycatch(Cvoid) do
        cpyjl_releasebuffer_impl(_o, _b)
    end Cvoid (CPyJlPtr{V}, Ptr{C.Py_buffer})

cpyjlattr(::Val{:__array_interface__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpycatch() do
            o = cpyjlvalue(_o)
            pydict(
                shape = size(o),
                typestr = "O",
                data = (UInt(pointer(o.ptrs)), false),
                strides = strides(o.ptrs) .* sizeof(CPyPtr),
                version = 3,
            )
        end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
    )

#### VECTOR, TUPLE, NAMED TUPLE as SEQUENCE

const SequenceLike = Union{AbstractVector, Tuple, NamedTuple}

pyjlabc(::Type{T}) where {T<:SequenceLike} = pycollectionsabcmodule.Sequence

cpyjlattr(::Val{:__getitem__}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
    if mighthavemethod(getindex, Tuple{T, Any})
        @cfunction (_o, _k) -> cpycatch(CPyPtr) do
            o = cpyjlvalue(_o)
            k = pynewobject(_k, true)
            i = pytryconvert(Union{Int,Tuple{Vararg{Int}}}, k)
            if i===PyConvertFail()
                pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
                return CPyPtr(0)
            end
            cpyreturn(CPyPtr, pyobject(o[(i.+1)...]))
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__setitem__}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
    if mighthavemethod(setindex!, Tuple{T, Any, Any})
        @cfunction (_o, _k, _v) -> cpycatch(Cint) do
            _v == C_NULL && error("deletion not implemented")
            o = cpyjlvalue(_o)
            k = pynewobject(_k, true)
            i = pytryconvert(Union{Int,Tuple{Vararg{Int}}}, k)
            if i===PyConvertFail()
                pyerrset(pyvalueerror, "invalid index of type '$(pytype(k).__name__)'")
                return -1
            end
            v = pytryconvert_value(o, pynewobject(_v, true), (i.+1)...)
            o[(i.+1)...] = v
            0
        end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)
    end

cpyjlattr(::Val{:index}, ::Type{T}, ::Type{V}) where {T<:Union{SequenceLike}, V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth => @cfunction (_o, _v) -> cpycatch() do
            o = cpyjlvalue(_o)
            v = pytryconvert_value(o, pynewobject(_v, true))
            i = v===PyConvertFail() ? nothing : findfirst(==(v), o)
            if i === nothing
                pyerrset(pyvalueerror, "no such value")
            else
                pyint(i - firstindex(o))
            end
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:count}, ::Type{T}, ::Type{V}) where {T<:Union{AbstractArray,SequenceLike}, V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth => @cfunction (_o, _v) -> cpycatch() do
            o = cpyjlvalue(_o)
            v = pytryconvert_value(o, pynewobject(_v, true))
            n = v===PyConvertFail() ? 0 : count(==(v), o)
            pyint(n)
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:append}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(push!, Tuple{T,Any})
        :method => Dict(
            :flags => C.Py_METH_O,
            :meth => @cfunction (_o, _v) -> cpycatch() do
                o = cpyjlvalue(_o)
                v = pytryconvert_value(o, pynewobject(_v, true))
                if v === PyConvertFail()
                    pyerrset(pytypeerror, "invalid element type")
                    CPyPtr(0)
                else
                    push!(o, v)
                    cpyreturn(CPyPtr, pynone)
                end
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:__iconcat__}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    @cfunction (_o, _v) -> cpycatch() do
        o = cpyjlvalue(_o)
        for _x in pynewobject(_v, true)
            x = pytryconvert_value(o, _x)
            if x === PyConvertFail()
                pyerrset(pytypeerror, "invalid element type")
                return CPyPtr(0)
            else
                push!(o, x)
            end
        end
        cpyreturn(CPyPtr, pynone)
    end CPyPtr (CPyJlPtr{V}, CPyPtr)

cpyjlattr(::Val{:extend}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(push!, Tuple{T,Any})
        :method => Dict(
            :flags => C.Py_METH_O,
            :meth => @cfunction (_o, _v) -> cpycatch() do
                pynewobject(_o, true).__iadd__(pynewobject(_v, true))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:insert}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(insert!, Tuple{T,Int,Any})
        :method => Dict(
            :flags => C.Py_METH_VARARGS,
            :meth => @cfunction (_o, _args) -> cpycatch() do
                o = cpyjlvalue(_o)
                i, _v = @pyconvert_args (i::Int, x) pynewobject(_args, true)
                v = pytryconvert_value(o, _v)
                if v === PyConvertFail()
                    pyerrset(pytypeerror, "invalid element type")
                    CPyPtr(0)
                else
                    insert!(o, i + firstindex(o), v)
                    cpyreturn(CPyPtr, pynone)
                end
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:pop}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(popat!, Tuple{T,Int})
        :method => Dict(
            :flags => C.Py_METH_VARARGS,
            :meth => @cfunction (_o, _args) -> cpycatch() do
                o = cpyjlvalue(_o)
                i, = @pyconvert_args (i::Union{Int,Nothing}=nothing,) pynewobject(_args, true)
                pyobject(popat!(o, i===nothing ? lastindex(o) : i + firstindex(o)))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:remove}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(popat!, Tuple{T,Int})
        :method => Dict(
            :flags => C.Py_METH_O,
            :meth => @cfunction (_o, _v) -> cpycatch() do
                o = pynewobject(_o, true)
                v = pynewobject(_v, true)
                o.pop(o.index(v))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:clear}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(empty!, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                o = cpyjlvalue(_o)
                empty!(o)
                pynone
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:sort}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(sort!, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_VARARGS | C.Py_METH_KEYWORDS,
            :meth => @cfunction (_o, _args, _kwargs) -> cpycatch() do
                o = cpyjlvalue(_o)
                key, rev = @pyconvert_args (key=pynone, rev::Bool=false) pynewobject(_args, true) (_kwargs == C_NULL ? pydict() : pynewobject(_kwargs, true))
                sort!(o; by = pyisnone(key) ? identity : key, rev = rev)
                pynone
            end CPyPtr (CPyJlPtr{V}, CPyPtr, CPyPtr)
        )
    end

cpyjlattr(::Val{:reverse}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(reverse!, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                o = cpyjlvalue(_o)
                reverse!(o)
                pynone
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:copy}, ::Type{T}, ::Type{V}) where {T<:SequenceLike, V} =
    if mighthavemethod(copy, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                o = cpyjlvalue(_o)
                pyjl(copy(o))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

#### SET AS SET

pyjlabc(::Type{T}) where {T<:AbstractSet} = pycollectionsabcmodule.Set

#### DICT AS MAPPING

pyjlabc(::Type{T}) where {T<:AbstractDict} = pycollectionsabcmodule.Mapping

#### NUMBER AS NUMBER

pyjlabc(::Type{T}) where {T<:Number} = pynumbersmodule.Number
pyjlabc(::Type{T}) where {T<:Complex} = pynumbersmodule.Complex
pyjlabc(::Type{T}) where {T<:Real} = pynumbersmodule.Real
pyjlabc(::Type{T}) where {T<:Rational} = pynumbersmodule.Rational
pyjlabc(::Type{T}) where {T<:Integer} = pynumbersmodule.Integral

#### IO AS IO

abstract type RawIO{V} <: SubClass{V} end
abstract type BufferedIO{V} <: SubClass{V} end
abstract type TextIO{V} <: SubClass{V} end

pyjlabc(::Type{T}) where {T<:IO} = pyiomodule.IOBase
pyjlabc(::Type{T}) where {T<:RawIO} = pyiomodule.RawIOBase
pyjlabc(::Type{T}) where {T<:BufferedIO} = pyiomodule.BufferedIOBase
pyjlabc(::Type{T}) where {T<:TextIO} = pyiomodule.TextIOBase

pyjlrawio(o::IO) = pyjl(o, RawIO{typeof(o)})
pyjlbufferedio(o::IO) = pyjl(o, BufferedIO{typeof(o)})
pyjltextio(o::IO) = pyjl(o, TextIO{typeof(o)})

# RawIO
# TODO:
# readline(size=-1)
# readlines(hint=-1)

cpyjlattr(::Val{:close}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> cpycatch() do
            close(cpyjlvalue(_o))
            pynone
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:closed}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpycatch() do
            pybool(!isopen(cpyjlvalue(_o)))
        end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
    )

cpyjlattr(::Val{:fileno}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(fd, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    pyint(fd(cpyjlvalue(_o)))
                end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "fileno"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:flush}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(flush, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    flush(cpyjlvalue(_o))
                    pynone
                end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "flush"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:isatty}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if T <: Base.TTY
                @cfunction (_o, _) -> cpyreturn(CPyPtr, pytrue) CPyPtr (CPyPtr, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> cpyreturn(CPyPtr, pyfalse) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:readable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(isreadable, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    pybool(isreadable(cpyjlvalue(_o)))
                end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readable"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:writable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(iswritable, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    pybool(iswritable(cpyjlvalue(_o)))
                end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "writable"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:tell}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(position, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    pyint(position(cpyjlvalue(_o)))
                end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "tell"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:writelines}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth => @cfunction (_o, _lines) -> cpycatch() do
            wr = pynewobject(_o, true).write
            for line in pynewobject(_lines, true)
                wr(line)
            end
            pyreturn(CPyPtr, pynone)
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:seekable}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(position, Tuple{T}) && mighthavemethod(seek, Tuple{T,Int}) && mighthavemethod(truncate, Tuple{T,Int})
                @cfunction (_o, _) -> cpyreturn(CPyPtr, pytrue) CPyPtr (CPyPtr, Ptr{Cvoid})
            else
                @cfunction (_o, _) -> cpyreturn(CPyPtr, pyfalse) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:truncate}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth =>
            if mighthavemethod(truncate, Tuple{T, Int}) && mighthavemethod(position, Tuple{T})
                @cfunction (_o, _args) -> cpycatch() do
                    n, = @pyconvert_args (size::Union{Int,Nothing}=nothing,) pynewobject(_args, true)
                    n === nothing && (n = position(o))
                    truncate(o, n)
                    pyint(n)
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "truncate"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:seek}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth =>
            if mighthavemethod(seek, Tuple{V, Int}) && mighthavemethod(position, Tuple{V})
                @cfunction (_o, _args) -> cpycatch() do
                    n, w = @pyconvert_args (offset::Int, whence::Int=0) pynewobject(_args, true)
                    o = cpyjlvalue(_o)
                    if w == 0
                        seek(o, n)
                    elseif w==1
                        seek(o, position(o)+n)
                    elseif w==2
                        seekend(o)
                        seek(o, position(o)+n)
                    else
                        pythrow(pyvalueerror("invalid whence: $w"))
                    end
                    pyint(position(o))
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "seek"); CPyPtr(0)) CPyPtr (CPyPtr, Ptr{Cvoid})
            end
    )

cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    @cfunction o -> (C.Py_IncRef(o); CPyPtr(o)) CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__next__}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    @cfunction _o -> cpycatch() do
        line = pynewobject(_o, true).readline()
        pylen(line) == 0 ? C_NULL : cpyreturn(CPyPtr, line)
    end CPyPtr (CPyJlPtr{V},)

# BufferedIO
# TODO:
# read1(size=-1)
# readinto1(b)

cpyjlattr(::Val{:detach}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "detach"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:readinto}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth =>
            if mighthavemethod(readbytes!, Tuple{V, Vector{UInt8}})
                @cfunction (_o, _b) -> cpycatch() do
                    b = PyBuffer(pynewobject(_b, true), C.PyBUF_WRITABLE)
                    o = cpyjlvalue(_o)
                    n = readbytes!(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
                    pyint(n)
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readinto"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
            end
    )

cpyjlattr(::Val{:read}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth =>
            if mighthavemethod(read, Tuple{V}) && mighthavemethod(read, Tuple{V, Int})
                @cfunction (_o, _args) -> cpycatch() do
                    n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pynewobject(_args, true)
                    o = cpyjlvalue(_o)
                    pybytes(convert(Vector{UInt8}, (n===nothing || n < 0) ? read(o) : read(o, n)))
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "read"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
            end
    )

cpyjlattr(::Val{:write}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth =>
            if mighthavemethod(write, Tuple{V, Vector{UInt8}})
                @cfunction (_o, _b) -> cpycatch() do
                    b = PyBuffer(pynewobject(_b, true), C.PyBUF_SIMPLE)
                    o = cpyjlvalue(_o)
                    n = write(o, unsafe_wrap(Array{UInt8}, Ptr{UInt8}(b.buf), b.len))
                    pyint(n)
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "write"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
            end
    )

cpyjlattr(::Val{:readline}, ::Type{BufferedIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth =>
            if mighthavemethod(read, Tuple{V, Type{UInt8}})
                @cfunction (_o, _args) -> cpycatch() do
                    n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pynewobject(_args, true)
                    o = cpyjlvalue(_o)
                    data = UInt8[]
                    while !eof(o) && (n===nothing || n < 0 || length(data) ≤ n)
                        c = read(o, UInt8)
                        push!(data, c)
                        c == 0x0A && break
                    end
                    pybytes(data)
                end CPyPtr (CPyJlPtr{V}, CPyPtr)
            else
                @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "readline"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
            end
    )

# TextIO

cpyjlattr(::Val{:encoding}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pystr("utf-8")) CPyPtr (CPyPtr, Ptr{Cvoid})
    )

cpyjlattr(::Val{:errors}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pystr("strict")) CPyPtr (CPyPtr, Ptr{Cvoid})
    )

cpyjlattr(::Val{:newlines}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpyreturn(CPyPtr, pynone) CPyPtr (CPyPtr, Ptr{Cvoid})
    )

cpyjlattr(::Val{:detach}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> (pyerrset(pyiounsupportedoperation, "detach"); CPyPtr(0)) CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:read}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth => @cfunction (_o, _args) -> cpycatch() do
            n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pynewobject(_args, true)
            o = cpyjlvalue(_o)
            b = IOBuffer()
            i = 0
            while !eof(o) && (n===nothing || n < 0 || i < n)
                i += 1
                write(b, read(o, Char))
            end
            seekstart(b)
            pystr(read(b, String))
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:readline}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_VARARGS,
        :meth => @cfunction (_o, _args) -> cpycatch() do
            n, = @pyconvert_args (size::Union{Int,Nothing}=-1,) pynewobject(_args, true)
            o = cpyjlvalue(_o)
            b = IOBuffer()
            i = 0
            while !eof(o) && (n===nothing || n < 0 || i < n)
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
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:write}, ::Type{TextIO{V}}, ::Type{V}) where {V} =
    :method => Dict(
        :flags => C.Py_METH_O,
        :meth => @cfunction (_o, _x) -> cpycatch() do
            x = pystr_asjuliastring(pynewobject(_x, true))
            o = cpyjlvalue(_o)
            n = 0
            linesep = pystr_asjuliastring(pyosmodule.linesep)
            for c in x
                if c == '\n'
                    write(o, linesep)
                else
                    write(o, c)
                end
                n += 1
            end
            pyint(n)
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )
