pyjulia_supertype(::Type{T}) where {T} = supertype(T)
pyjulia_isconcrete(::Type{T}) where {T} = isconcretetype(T)
pyjulia_unwrappedtype(::Type{T}) where {T} = T
pyjulia_unwrap(x) = x

@kwdef struct CPyJuliaTypeObject{T} <: AbstractCPyTypeObject
    ob_base :: CPyTypeObject = CPyTypeObject()
end

@kwdef struct CPyJuliaObject{T} <: AbstractCPyObject
    ob_base :: CPyObject = CPyObject()
    value :: Ptr{Cvoid} = C_NULL
    weaklist :: CPyPtr = C_NULL
end

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

    # number methods
    nb_opts = Dict(
        :bool => cpyjlspecialattr(Val(:__bool__), T),
        :int => cpyjlspecialattr(Val(:__int__), T),
        :float => cpyjlspecialattr(Val(:__float__), T),
        :index => cpyjlspecialattr(Val(:__index__), T),
        :negative => cpyjlspecialattr(Val(:__neg__), T),
        :positive => cpyjlspecialattr(Val(:__pos__), T),
        :absolute => cpyjlspecialattr(Val(:__abs__), T),
        :invert => cpyjlspecialattr(Val(:__invert__), T),
    )
    filter!(x -> x.second != C_NULL, nb_opts)
    isempty(nb_opts) && (nb_opts = C_NULL)

    # mapping methods
    mp_opts = Dict(
        :length => cpyjlspecialattr(Val(:__len__), T),
        :subscript => cpyjlspecialattr(Val(:__getitem__), T),
        :ass_subscript => cpyjlspecialattr(Val(:__setitem__), T),
    )
    filter!(x -> x.second != C_NULL, mp_opts)
    isempty(mp_opts) && (mp_opts = C_NULL)

    # sequence methods
    sq_opts = Dict(
        :length => cpyjlspecialattr(Val(:__len__), T),
        :item => cpyjlspecialattr(Val(:__getitem_int__), T),
        :ass_item => cpyjlspecialattr(Val(:__setitem_int__), T),
        :contains => cpyjlspecialattr(Val(:__contains__), T),
        :concat => cpyjlspecialattr(Val(:__concat__), T),
        :inplace_concat => cpyjlspecialattr(Val(:__iconcat__), T),
        :repeat => cpyjlspecialattr(Val(:__repeat__), T),
        :inplace_repeat => cpyjlspecialattr(Val(:__irepeat__), T),
    )
    filter!(x -> x.second != C_NULL, sq_opts)
    isempty(sq_opts) && (sq_opts = C_NULL)

    # buffer methods
    buf_opts = Dict(
        :get => cpyjlspecialattr(Val(:__getbuffer__), T),
        :release => cpyjlspecialattr(Val(:__releasebuffer__), T),
    )
    filter!(x -> x.second != C_NULL, buf_opts)
    isempty(buf_opts) && (buf_opts = C_NULL)

    # type options
    opts = Dict{Symbol, Any}(
        :name => "julia.$T",
        :base => T==Any ? pyobjecttype : T==DataType ? pyjuliatype(Type) : pyjuliatype(pyjulia_supertype(T)),
        :flags => CPy_TPFLAGS_BASETYPE | CPy_TPFLAGS_HAVE_VERSION_TAG | (PYISSTACKLESS ? CPy_TPFLAGS_HAVE_STACKLESS_EXTENSION : 0x00),
        :basicsize => sizeof(CPyJuliaObject{T}),
        :dealloc => cpyjlspecialattr(Val(:__dealloc__), T),
        :hash => cpyjlspecialattr(Val(:__hash__), T),
        :repr => cpyjlspecialattr(Val(:__repr__), T),
        :str => cpyjlspecialattr(Val(:__str__), T),
        :iter => cpyjlspecialattr(Val(:__iter__), T),
        :iternext => cpyjlspecialattr(Val(:__next__), T),
        :getattr => cpyjlspecialattr(Val(:__getattr_str__), T),
        :setattr => cpyjlspecialattr(Val(:__setattr_str__), T),
        :getattro => cpyjlspecialattr(Val(:__getattr__), T),
        :setattro => cpyjlspecialattr(Val(:__setattr__), T),
        :as_number => nb_opts,
        :as_mapping => mp_opts,
        :as_sequence => sq_opts,
        :as_buffer => buf_opts,
    )

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

    # TODO: register ABCs

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

function pyjulia(x::T) where {T}
    # get the type of the object
    t = pyjuliatype(T)
    # allocate an object
    o = cpycall_obj(Val(:_PyObject_New), t)
    # set weakrefs and value
    p = UnsafePtr{CPyJuliaObject{T}}(pyptr(o))
    p.weaklist[] = C_NULL
    p.value[], PYJLGCCACHE[pyptr(o)] = pointer_from_obj(pyjulia_unwrap(x))
    # done
    return o
end
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

cpyjuliavalue(p::Ptr{CPyJuliaObject{T}}) where {T} = Base.unsafe_pointer_to_objref(UnsafePtr(p).value[!]) :: pyjulia_unwrappedtype(T)
cpyjuliavalue(p::Ptr) = cpyjuliavalue(Ptr{CPyJuliaObject{Any}}(p))

function cpyjlspecialattr(::Val{k}, ::Type{T}) where {k, T}
    a = cpyjlattr(Val(k), T)
    a isa Ptr ? a : C_NULL
end

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
# ATTRIBUTE DEFINITIONS

cpyjlattr(::Val, ::Type) = nothing

cpyjlattr(::Val{:__dealloc__}, ::Type{T}) where {T} =
    @cfunction function (_o)
        UnsafePtr(_o).weaklist[!] == C_NULL || cpycall_voidx(Val(:PyObject_ClearWeakRefs), _o)
        delete!(PYJLGCCACHE, CPyPtr(_o))
        nothing
    end Cvoid (Ptr{CPyJuliaObject{T}},)

cpyjlattr(::Val{:__repr__}, ::Type{T}) where {T} =
    @cfunction _o -> cpycatch() do
        pystr(string("<jl ", repr(cpyjuliavalue(_o)), ">"))
    end CPyPtr (Ptr{CPyJuliaObject{T}},)

cpyjlattr(::Val{:__str__}, ::Type{T}) where {T} =
    @cfunction _o -> cpycatch() do
        pystr(string(cpyjuliavalue(_o)))
    end CPyPtr (Ptr{CPyJuliaObject{T}},)

cpyjlattr(::Val{:__len__}, ::Type{T}) where {T} =
    if hasmethod(length, Tuple{T})
        @cfunction _o -> cpycatch(CPy_ssize_t) do
            length(cpyjuliavalue(_o))
        end CPy_ssize_t (Ptr{CPyJuliaObject{T}},)
    end

cpyjlattr(::Val{:__hash__}, ::Type{T}) where {T} =
    if hasmethod(hash, Tuple{T})
        @cfunction _o -> cpycatch(CPy_hash_t) do
            h = hash(cpyjuliavalue(_o)) % CPy_hash_t
            h == zero(CPy_hash_t)-one(CPy_hash_t) ? zero(CPy_hash_t) : h
        end CPy_hash_t (Ptr{CPyJuliaObject{T}},)
    end

mutable struct Iterator{T}
    val :: T
    st :: Union{Nothing, Some}
end

cpyjlattr(::Val{:__iter__}, ::Type{T}) where {T} =
    if hasmethod(iterate, Tuple{T})
        @cfunction _o -> cpycatch() do
            pyjulia(Iterator(cpyjuliavalue(_o), nothing))
        end CPyPtr (Ptr{CPyJuliaObject{T}},)
    end

cpyjlattr(::Val{:__iter__}, ::Type{T}) where {T<:Iterator} =
    @cfunction _o->(cpyincref(_o); CPyPtr(_o)) CPyPtr (Ptr{CPyJuliaObject{T}},)

cpyjlattr(::Val{:__next__}, ::Type{T}) where {T<:Iterator} =
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
    end  CPyPtr (Ptr{CPyJuliaObject{T}},)

pyjl_attrname_py2jl(x::AbstractString) =
    replace(x, r"_[_b]" => s -> (s[2]=='b' ? '!' : '_'))

pyjl_attrname_jl2py(x::AbstractString) =
    replace(replace(x, r"_(?=[_b])" => "__"), '!'=>"_b")

cpyjlattr(::Val{:__getattr__}, ::Type{T}) where {T} =
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
    end CPyPtr (Ptr{CPyJuliaObject{T}}, CPyPtr)

pyjlisbufferabletype(::Type{T}) where {T} =
    T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Bool, Ptr{Cvoid})
pyjlisbufferabletype(::Type{T}) where {T<:Tuple} =
    isconcretetype(T) && Base.allocatedinline(T) && all(pyjlisbufferabletype, fieldtypes(T))
pyjlisbufferabletype(::Type{NamedTuple{names,T}}) where {names,T} =
    pyjlisbufferabletype(T)

function ismutablearray(x::AbstractArray)
    try
        i = firstindex(x)
        y = x[i]
        x[i] = y
        true
    catch
        false
    end
end

pybufferformat(::Type{T}) where {T} =
    T == Int8 ? "=b" :
    T == UInt8 ? "=B" :
    T == Int16 ? "=h" :
    T == UInt16 ? "=H" :
    T == Int32 ? "=i" :
    T == UInt32 ? "=I" :
    T == Int64 ? "=q" :
    T == UInt64 ? "=Q" :
    T == Float16 ? "=e" :
    T == Float32 ? "=f" :
    T == Float64 ? "=d" :
    T == Complex{Float16} ? "=Ze" :
    T == Complex{Float32} ? "=Zf" :
    T == Complex{Float64} ? "=Zd" :
    T == Bool ? "?" :
    T == Ptr{Cvoid} ? "P" :
    if isstructtype(T) && isconcretetype(T) && Base.allocatedinline(T)
        n = fieldcount(T)
        flds = []
        for i in 1:n
            nm = fieldname(T, i)
            tp = fieldtype(T, i)
            push!(flds, string(pybufferformat(tp), nm isa Symbol ? ":$nm:" : ""))
            d = (i==n ? sizeof(T) : fieldoffset(T, i+1)) - (fieldoffset(T, i) + sizeof(tp))
            @assert d≥0
            d>0 && push!(flds, "$(d)x")
        end
        string("T{", join(flds, " "), "}")
    else
        "$(Base.aligned_sizeof(T))x"
    end

cpyjlattr(::Val{:__getbuffer__}, ::Type{A}) where {T, A<:AbstractArray{T}} =
    if hasmethod(pointer, Tuple{A}) && hasmethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b, flags) -> cpycatch(Cint) do
            o = cpyjuliavalue(_o)
            b = UnsafePtr(_b)
            c = []
            b.obj[] = C_NULL

            # buf
            b.buf[] = pointer(o)

            # readonly
            b.readonly[] = ismutablearray(o) ? 0 : 1

            # itemsize
            b.itemsize[] = elsz = Base.aligned_sizeof(eltype(o))

            # len
            b.len[] = elsz * length(o)

            # format
            if iszero(flags & CPyBUF_FORMAT)
                b.format[] = C_NULL
            else
                b.format[] = cacheptr!(c, pybufferformat(eltype(o)))
            end

            # ndim
            b.ndim[] = ndims(o)

            # shape
            if iszero(flags & CPyBUF_ND)
                b.shape[] = C_NULL
            else
                b.shape[] = cacheptr!(c, CPy_ssize_t[size(o)...])
            end

            # strides
            if iszero(flags & CPyBUF_STRIDES)
                b.strides[] = C_NULL
            else
                b.strides[] = cacheptr!(c, CPy_ssize_t[(strides(o) .* elsz)...])
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
        end Cint (Ptr{CPyJuliaObject{A}}, Ptr{CPy_buffer}, Cint)
    end

cpyjlattr(::Val{:__releasebuffer__}, ::Type{A}) where {T, A<:AbstractArray{T}} =
    if hasmethod(pointer, Tuple{A}) && hasmethod(strides, Tuple{A}) && pyjlisbufferabletype(T)
        @cfunction (_o, _b) -> cpycatch(Cvoid) do
            b = UnsafePtr(_b)
            delete!(PYJLBUFCACHE, b.internal[])
            nothing
        end Cvoid (Ptr{CPyJuliaObject{A}}, Ptr{CPy_buffer})
    end
