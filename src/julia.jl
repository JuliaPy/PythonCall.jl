mighthavemethod(f, t) = !isempty(methods(f, t))

abstract type SubClass{V} end

pyjulia_supertype(::Type{T}) where {T} = supertype(T)
pyjulia_supertype(::Type{Any}) = nothing
pyjulia_supertype(::Type{DataType}) = Type
pyjulia_supertype(::Type{T}) where {V, T<:SubClass{V}} =
    if T == SubClass{V}
        error()
    elseif supertype(T) == SubClass{V}
        V
    else
        supertype(T)
    end

pyjulia_valuetype(::Type{T}) where {T} = T
pyjulia_valuetype(::Type{T}) where {V, T<:SubClass{V}} = V

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

const pyjuliaexception = PyLazyObject() do
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
export pyjuliaexception

function pyjuliatype(::Type{T}) where {T}
    # see if we already made this type
    r = get(PYJLTYPES, T, nothing)
    r === nothing || return r

    # get attributes
    V = pyjulia_valuetype(T)
    S = T
    attrnames = Set{Symbol}()
    special = Dict{Symbol, Ptr{Cvoid}}()
    methods = []
    getset = []
    while true
        V <: pyjulia_valuetype(S) || error("value type of $T is $V, but value type of supertype $S is $(pyjulia_valuetype(S))")
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
        S = pyjulia_supertype(S)
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
        :base => pyjulia_supertype(T) === nothing ? pyobjecttype : pyjuliatype(pyjulia_supertype(T)),
        :flags => C.Py_TPFLAGS_BASETYPE | C.Py_TPFLAGS_HAVE_VERSION_TAG | (PYISSTACKLESS ? C.Py_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
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
export pyjuliatype

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

function pyjulia(x, ::Type{T}) where {T}
    x isa pyjulia_valuetype(T) || error("x must be a `$(pyjulia_valuetype(T))`")
    # get the type of the object
    t = pyjuliatype(T)
    # allocate an object
    o = check(C._PyObject_New(t))
    # set weakrefs and value
    p = UnsafePtr{CPyJuliaObject{T}}(pyptr(o))
    p.weaklist[] = C_NULL
    p.value[], PYJLGCCACHE[pyptr(o)] = pointer_from_obj(x)
    # done
    return o
end
pyjulia(x) = pyjulia(x, typeof(x))
export pyjulia

pyisjulia(o::AbstractPyObject) = pytypecheck(o, pyjuliatype(Any))
export pyisjulia

function pyjuliavalue(o::AbstractPyObject)
    if pyisjulia(o)
        GC.@preserve o cpyjuliavalue(pyptr(o))
    else
        pythrow(pytypeerror("not a julia object"))
    end
end
export pyjuliavalue

cpyjuliavalue(p::Ptr{CPyJuliaObject{T}}) where {T} = Base.unsafe_pointer_to_objref(UnsafePtr(p).value[!]) :: pyjulia_valuetype(T)
cpyjuliavalue(p::Ptr) = cpyjuliavalue(Ptr{CPyJuliaObject{Any}}(p))

function cpycatch(f, ::Type{T}=CPyPtr) where {T}
    try
        cpyreturn(T, f())
    catch err
        if err isa PythonRuntimeError
            # We restore Python errors.
            # TODO: Is this the right behaviour?
            C.PyErr_Restore(pyincref!(err.t), pyincref!(err.v), pyincref!(err.b))
        else
            # Other (Julia) errors are raised as a JuliaException
            bt = catch_backtrace()
            val = try
                pyjulia((err, bt))
            catch
                pynone
            end
            pyerrset(pyjuliaexception, val)
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
        pystr(string("<jl ", repr(cpyjuliavalue(_o)), ">"))
    end CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__str__}, ::Type{T}, ::Type{V}) where {T, V} =
    @cfunction _o -> cpycatch() do
        pystr(string(cpyjuliavalue(_o)))
    end CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__len__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(length, Tuple{T})
        @cfunction _o -> cpycatch(C.Py_ssize_t) do
            length(cpyjuliavalue(_o))
        end C.Py_ssize_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__hash__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(hash, Tuple{T})
        @cfunction _o -> cpycatch(C.Py_hash_t) do
            h = hash(cpyjuliavalue(_o)) % C.Py_hash_t
            h == zero(C.Py_hash_t)-one(C.Py_hash_t) ? zero(C.Py_hash_t) : h
        end C.Py_hash_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__contains__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(in, Tuple{Any, T})
        @cfunction (_o, _v) -> cpycatch(Cint) do
            o = cpyjuliavalue(_o)
            v = pytryconvert_element(o, pynewobject(_v, true))
            v === PyConvertFail() ? false : in(v, o)
        end Cint (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__reversed__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(reverse, Tuple{T})
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch(CPyPtr) do
                pyjulia(reverse(cpyjuliavalue(_o)))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:__getitem__}, ::Type{T}, ::Type{V}) where {T, V} =
    if mighthavemethod(getindex, Tuple{T, Any})
        @cfunction (_o, _k) -> cpycatch(CPyPtr) do
            o = cpyjuliavalue(_o)
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
            o = cpyjuliavalue(_o)
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
            pyjulia(Iterator(cpyjuliavalue(_o), nothing))
        end CPyPtr (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T<:Iterator, V} =
    @cfunction _o->(cpyincref(_o); CPyPtr(_o)) CPyPtr (CPyJlPtr{V},)

cpyjlattr(::Val{:__next__}, ::Type{T}, ::Type{V}) where {T<:Iterator, V} =
    @cfunction _o -> cpycatch() do
        o = cpyjuliavalue(_o)
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
        # then see if there is a corresponding julia property
        o = cpyjuliavalue(_o)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pynewobject(_k, true))))
        if hasproperty(o, k)
            pyerrclear() # the attribute error is still set
            return cpyreturn(CPyPtr, pyobject(getproperty(o, k)))
        end
        # no such attribute
        return _x
    end CPyPtr (CPyJlPtr{V}, CPyPtr)

cpyjlattr(::Val{:__setattr__}, ::Type{T}, ::Type{V}) where {T,V} =
    @cfunction (_o, _k, _v) -> cpycatch(Cint) do
        # first do the generic version
        err = C.PyObject_GenericSetAttr(_o, _k, _v)
        (err == -1 && pyerroccurred(pyattributeerror)) || return err
        # now see if there is a corresponding julia property
        _v == C_NULL && error("deletion not supported")
        o = cpyjuliavalue(_o)
        k = Symbol(pyjl_attrname_py2jl(pystr_asjuliastring(pynewobject(_k, true))))
        if hasproperty(o, k)
            pyerrclear() # the attribute error is still set
            v = pyconvert(Any, pynewobject(_v, true)) # can we do better than Any?
            @show v typeof(v)
            setproperty!(o, k, v)
            return Cint(0)
        end
        # no such attribute
        return err
    end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)

cpyjlattr(::Val{:__dir__}, ::Type{T}, ::Type{V}) where {T, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> cpycatch() do
            # default properties
            r = pyobjecttype.__dir__(pynewobject(_o, true))
            # add julia properties
            o = cpyjuliavalue(_o)
            for k in propertynames(o)
                r.append(pystr(k))
            end
            r
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:__call__}, ::Type{T}, ::Type{V}) where {T,V} =
    @cfunction (_o, _args, _kwargs) -> cpycatch() do
        o = cpyjuliavalue(_o)
        args = [pyconvert(Any, v) for v in pynewobject(_args, true)]
        kwargs = _kwargs==C_NULL ? Dict() : Dict(Symbol(pystr_asjuliastring(k)) => pyconvert(Any, v) for (k,v) in pynewobject(_kwargs, true))
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
    if isflagset(flags, CPyBUF_WRITABLE)
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
    if isflagset(flags, CPyBUF_FORMAT)
        b.format[] = cacheptr!(c, fmt)
    else
        b.format[] = C_NULL
    end

    # shape
    if isflagset(flags, CPyBUF_ND)
        b.shape[] = cacheptr!(c, CPy_ssize_t[sz...])
    else
        b.shape[] = C_NULL
    end

    # strides
    if isflagset(flags, CPyBUF_STRIDES)
        b.strides[] = cacheptr!(c, CPy_ssize_t[(strds .* elsz)...])
    else
        if size_to_cstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not C contiguous and strides not requested")
            return -1
        end
        b.strides[] = C_NULL
    end

    # check contiguity
    if isflagset(flags, CPyBUF_C_CONTIGUOUS)
        if size_to_cstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not C contiguous")
            return -1
        end
    end
    if isflagset(flags, CPyBUF_F_CONTIGUOUS)
        if size_to_fstrides(1, sz...) != strds
            pyerrset(pybuffererror, "not Fortran contiguous")
            return -1
        end
    end
    if isflagset(flags, CPyBUF_ANY_CONTIGUOUS)
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
    cpyincref(_o)
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
            o = cpyjuliavalue(_o)
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
                o = cpyjuliavalue(_o)
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
    if mighthavemethod(pointer, Tuple{A}) && mighthavemethod(strides, Tuple{A}) && pyjlisarrayabletype(T)
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                pyimport("numpy").asarray(cpyjuliavalue(_o))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    else
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                pyimport("numpy").asarray(pyjulia(PyObjectArray(cpyjuliavalue(_o))))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

#### PYOBJECTARRAY AS BUFFER AND ARRAY

cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    @cfunction (_o, _b, flags) -> cpycatch(Cint) do
        o = cpyjuliavalue(_o)
        cpyjl_getbuffer_impl(_o, _b, flags, pointer(o.ptrs), sizeof(CPyPtr), length(o), ndims(o), "O", size(o), strides(o.ptrs), true)
    end Cint (CPyJlPtr{V}, Ptr{C.Py_buffer}, Cint)

cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    @cfunction (_o, _b) -> cpycatch(Cvoid) do
        cpyjl_releasebuffer_impl(_o, _b)
    end Cvoid (CPyJlPtr{V}, Ptr{C.Py_buffer})

cpyjlattr(::Val{:__array_interface__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpycatch() do
            o = cpyjuliavalue(_o)
            pydict(
                shape = size(o),
                typestr = "O",
                data = (UInt(pointer(o.ptrs)), false),
                strides = strides(o.ptrs) .* sizeof(CPyPtr),
                version = 3,
            )
        end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
    )

cpyjlattr(::Val{:__array__}, ::Type{A}, ::Type{V}) where {A<:PyObjectArray, V} =
        :method => Dict(
            :flags => C.Py_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch() do
                pyimport("numpy").asarray(cpyjuliavalue(_o))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )

#### VECTOR AND TUPLE AS SEQUENCE

pyjlabc(::Type{T}) where {T<:AbstractVector} = pyimport("collections.abc").Sequence
pyjlabc(::Type{T}) where {T<:Tuple} = pyimport("collections.abc").Sequence

#### SET AS SET

pyjlabc(::Type{T}) where {T<:AbstractSet} = pyimport("collections.abc").Set

#### DICT AS MAPPING

pyjlabc(::Type{T}) where {T<:AbstractDict} = pyimport("collections.abc").Mapping

#### NUMBER AS NUMBER

pyjlabc(::Type{T}) where {T<:Number} = pyimport("numbers").Number
pyjlabc(::Type{T}) where {T<:Complex} = pyimport("numbers").Complex
pyjlabc(::Type{T}) where {T<:Real} = pyimport("numbers").Real
pyjlabc(::Type{T}) where {T<:Rational} = pyimport("numbers").Rational
pyjlabc(::Type{T}) where {T<:Integer} = pyimport("numbers").Integral

#### IO AS IO

abstract type RawIO{V} <: SubClass{V} end
abstract type BufferedIO{V} <: SubClass{V} end
abstract type TextIO{V} <: SubClass{V} end

pyjlabc(::Type{T}) where {T<:IO} = pyimport("io").IOBase
pyjlabc(::Type{T}) where {T<:RawIO} = pyimport("io").RawIOBase
pyjlabc(::Type{T}) where {T<:BufferedIO} = pyimport("io").BufferedIOBase
pyjlabc(::Type{T}) where {T<:TextIO} = pyimport("io").TextIOBase

pyjuliarawio(o::IO) = pyjulia(o, RawIO{typeof(o)})
pyjuliabufferedio(o::IO) = pyjulia(o, BufferedIO{typeof(o)})
pyjuliatextio(o::IO) = pyjulia(o, TextIO{typeof(o)})

# RawIO
# TODO:
# readline(size=-1)
# readlines(hint=-1)

cpyjlattr(::Val{:close}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth => @cfunction (_o, _) -> cpycatch() do
            close(cpyjuliavalue(_o))
            pynone
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    )

cpyjlattr(::Val{:closed}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :property => Dict(
        :get => @cfunction (_o, _) -> cpycatch() do
            pybool(!isopen(cpyjuliavalue(_o)))
        end CPyPtr (CPyJlPtr{V}, Ptr{Cvoid})
    )

cpyjlattr(::Val{:fileno}, ::Type{T}, ::Type{V}) where {T<:IO, V} =
    :method => Dict(
        :flags => C.Py_METH_NOARGS,
        :meth =>
            if mighthavemethod(fd, Tuple{T})
                @cfunction (_o, _) -> cpycatch() do
                    pyint(fd(cpyjuliavalue(_o)))
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
                    flush(cpyjuliavalue(_o))
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
                    pybool(isreadable(cpyjuliavalue(_o)))
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
                    pybool(iswritable(cpyjuliavalue(_o)))
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
                    pyint(position(cpyjuliavalue(_o)))
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
                    args = pyconvert(Union{Tuple, Tuple{Union{Int,Nothing}}}, pynewobject(_args, true))
                    o = cpyjuliavalue(_o)
                    n = (args===() || args===(nothing,)) ? position(o) : args[1]
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
                    args = pyconvert(Union{Tuple{Int}, Tuple{Int,Int}}, pynewobject(_args, true))
                    o = cpyjuliavalue(_o)
                    if length(args)==1 || args[2]==0
                        seek(o, args[1])
                    elseif args[2]==1
                        seek(o, position(o)+args[1])
                    elseif args[2]==2
                        seekend(o)
                        seek(o, position(o)+args[1])
                    else
                        error("invalid whence: $(args[2])")
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
                    o = cpyjuliavalue(_o)
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
                    args = pyconvert(Union{Tuple{}, Tuple{Int}}, pynewobject(_args, true))
                    o = cpyjuliavalue(_o)
                    pybytes(convert(Vector{UInt8}, length(args)==0 ? read(o) : read(o, args[1])))
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
                    o = cpyjuliavalue(_o)
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
                    args = pyconvert(Union{Tuple{}, Tuple{Int}}, pynewobject(_args, true))
                    o = cpyjuliavalue(_o)
                    data = UInt8[]
                    while !eof(o) && (args===() || length(data) ≤ args[1])
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
