pyjulia_supertype(::Type{T}) where {T} = supertype(T)
pyjulia_supertype(::Type{Any}) = nothing
pyjulia_supertype(::Type{DataType}) = Type

pyjulia_valuetype(::Type{T}) where {T} = T

@kwdef struct CPyJuliaTypeObject{T} <: AbstractCPyTypeObject
    ob_base :: CPyTypeObject = CPyTypeObject()
end

@kwdef struct CPyJuliaObject{T} <: AbstractCPyObject
    ob_base :: CPyObject = CPyObject()
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
    cpycall_void(Val(:PyType_Ready), ptr)
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
        :flags => CPy_TPFLAGS_BASETYPE | CPy_TPFLAGS_HAVE_VERSION_TAG | (PYISSTACKLESS ? CPy_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
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
    cpycall_void(Val(:PyType_Ready), ptr)

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
    o = cpycall_obj(Val(:_PyObject_New), t)
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
        bt = catch_backtrace()
        val = try
            pyjulia((err, bt))
        catch
            pynone
        end
        pyerrset(pyjuliaexception, val)
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

pyjlabc(::Type{T}) where {T<:AbstractVector} = pyimport("collections.abc").Sequence
pyjlabc(::Type{T}) where {T<:AbstractSet} = pyimport("collections.abc").Set
pyjlabc(::Type{T}) where {T<:AbstractDict} = pyimport("collections.abc").Mapping

pyjlabc(::Type{T}) where {T<:Number} = pyimport("numbers").Number
pyjlabc(::Type{T}) where {T<:Complex} = pyimport("numbers").Complex
pyjlabc(::Type{T}) where {T<:Real} = pyimport("numbers").Real
pyjlabc(::Type{T}) where {T<:Rational} = pyimport("numbers").Rational
pyjlabc(::Type{T}) where {T<:Integer} = pyimport("numbers").Integral

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
        UnsafePtr(_o).weaklist[!] == C_NULL || cpycall_voidx(Val(:PyObject_ClearWeakRefs), _o)
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
    if hasmethod(length, Tuple{T})
        @cfunction _o -> cpycatch(CPy_ssize_t) do
            length(cpyjuliavalue(_o))
        end CPy_ssize_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__hash__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(hash, Tuple{T})
        @cfunction _o -> cpycatch(CPy_hash_t) do
            h = hash(cpyjuliavalue(_o)) % CPy_hash_t
            h == zero(CPy_hash_t)-one(CPy_hash_t) ? zero(CPy_hash_t) : h
        end CPy_hash_t (CPyJlPtr{V},)
    end

cpyjlattr(::Val{:__contains__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(in, Tuple{Any, T})
        @cfunction (_o, _v) -> cpycatch(Cint) do
            o = cpyjuliavalue(_o)
            v = pytryconvert_element(o, pynewobject(_v, true))
            v === PyConvertFail() ? false : in(v, o)
        end Cint (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__reversed__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(reverse, Tuple{T})
        :method => Dict(
            :flags => CPy_METH_NOARGS,
            :meth => @cfunction (_o, _) -> cpycatch(CPyPtr) do
                pyjulia(reverse(cpyjuliavalue(_o)))
            end CPyPtr (CPyJlPtr{V}, CPyPtr)
        )
    end

cpyjlattr(::Val{:__getitem__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(getindex, Tuple{T, Any})
        @cfunction (_o, _k) -> cpycatch(CPyPtr) do
            o = cpyjuliavalue(_o)
            k = pyconvert_key(o, pynewobject(_k, true))
            pyobject(o[k])
        end CPyPtr (CPyJlPtr{V}, CPyPtr)
    end

cpyjlattr(::Val{:__setitem__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(setindex!, Tuple{T, Any, Any})
        @cfunction (_o, _k, _v) -> cpycatch(Cint) do
            _v == C_NULL && error("deletion not implemented")
            o = cpyjuliavalue(_o)
            k = pyconvert_key(o, pynewobject(_k, true))
            v = pyconvert_value(o, k, pynewobject(_v, true))
            o[k] = v
            0
        end Cint (CPyJlPtr{V}, CPyPtr, CPyPtr)
    end

mutable struct Iterator{T}
    val :: T
    st :: Union{Nothing, Some}
end

cpyjlattr(::Val{:__iter__}, ::Type{T}, ::Type{V}) where {T, V} =
    if hasmethod(iterate, Tuple{T})
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
        _x = cpycall_raw(Val(:PyObject_GenericGetAttr), CPyPtr, _o, _k)
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

pyjlisbufferabletype(::Type{T}) where {T} =
    T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Bool, Ptr{Cvoid})
pyjlisbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) && Base.allocatedinline(T) && all(pyjlisbufferabletype, fieldtypes(T))
pyjlisbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjlisbufferabletype(T)

cpyjlattr(::Val{:__getbuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    if hasmethod(pointer, Tuple{A}) && hasmethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b, flags) -> cpycatch(Cint) do
            o = cpyjuliavalue(_o)
            b = UnsafePtr(_b)
            c = []

            # not influenced by flags: obj, buf, len, itemsize, ndim
            b.obj[] = C_NULL
            b.buf[] = pointer(o)
            b.itemsize[] = elsz = Base.aligned_sizeof(eltype(o))
            b.len[] = elsz * length(o)
            b.ndim[] = ndims(o)

            # readonly
            if !iszero(flags & CPyBUF_WRITABLE)
                if ismutablearray(o)
                    b.readonly[] = 1
                else
                    pyerrset(pybuffererror, "not writable")
                    return -1
                end
            else
                b.readonly[] = ismutablearray(o) ? 0 : 1
            end

            # format
            if !iszero(flags & CPyBUF_FORMAT)
                b.format[] = cacheptr!(c, pybufferformat(eltype(o)))
            else
                b.format[] = C_NULL
            end

            # shape
            if !iszero(flags & CPyBUF_ND)
                b.shape[] = cacheptr!(c, CPy_ssize_t[size(o)...])
            else
                b.shape[] = C_NULL
            end

            # strides
            if !iszero(flags & CPyBUF_STRIDES)
                b.strides[] = cacheptr!(c, CPy_ssize_t[(strides(o) .* elsz)...])
            else
                if !isccontiguous(o)
                    pyerrset(pybuffererror, "not C contiguous")
                    return -1
                end
                b.strides[] = C_NULL
            end

            # check contiguity
            if !iszero(flags & CPyBUF_C_CONTIGUOUS)
                if !isccontiguous(o)
                    pyerrset(pybuffererror, "not C contiguous")
                    return -1
                end
            end
            if !iszero(flags & CPyBUF_F_CONTIGUOUS)
                if !isfcontiguous(o)
                    pyerrset(pybuffererror, "not Fortran contiguous")
                    return -1
                end
            end
            if !iszero(flags & CPyBUF_ANY_CONTIGUOUS)
                if !isccontiguous(o) && !isfcontiguous(o)
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
        end Cint (CPyJlPtr{V}, Ptr{CPy_buffer}, Cint)
    end

cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}, ::Type{V}) where {T, A<:AbstractArray{T}, V} =
    if hasmethod(pointer, Tuple{A}) && hasmethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b) -> cpycatch(Cvoid) do
            b = UnsafePtr(_b)
            delete!(PYJLBUFCACHE, b.internal[])
            nothing
        end Cvoid (CPyJlPtr{V}, Ptr{CPy_buffer})
    end
