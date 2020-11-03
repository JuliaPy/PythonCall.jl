abstract type AbstractCPyObject end

@kwdef struct CPyObject
    # assumes _PyObject_HEAD_EXTRA is empty
    refcnt :: CPy_ssize_t = 0
    type :: Ptr{CPyObject} = C_NULL
end

const CPyPtr = Ptr{CPyObject}

cpyrefcnt(o::Ptr) = unsafe_load(CPyPtr(o)).refcnt
cpytype(o::Ptr) = unsafe_load(CPyPtr(o)).type
cpyincref(o::Ptr) = cpycall_voidx(Val(:Py_IncRef), o)
cpydecref(o::Ptr) = cpycall_voidx(Val(:Py_DecRef), o)

abstract type AbstractCPyVarObject end

@kwdef struct CPyVarObject
    ob_base :: CPyObject = CPyObject()
    size :: CPy_ssize_t = 0
end

abstract type AbstractCPyTypeObject end

@kwdef struct CPyTypeObject <: AbstractCPyTypeObject
    ob_base :: CPyVarObject = CPyVarObject()
    name :: Cstring = C_NULL

    basicsize :: CPy_ssize_t = 0
    itemsize :: CPy_ssize_t = 0

    dealloc :: Ptr{Cvoid} = C_NULL
    vectorcall_offset :: CPy_ssize_t = C_NULL
    getattr :: Ptr{Cvoid} = C_NULL
    setattr :: Ptr{Cvoid} = C_NULL
    as_async :: Ptr{Cvoid} = C_NULL
    repr :: Ptr{Cvoid} = C_NULL

    as_number :: Ptr{CPyNumberMethodsStruct} = C_NULL
    as_sequence :: Ptr{CPySequenceMethodsStruct} = C_NULL
    as_mapping :: Ptr{CPyMappingMethodsStruct} = C_NULL

    hash :: Ptr{Cvoid} = C_NULL
    call :: Ptr{Cvoid} = C_NULL
    str :: Ptr{Cvoid} = C_NULL
    getattro :: Ptr{Cvoid} = C_NULL
    setattro :: Ptr{Cvoid} = C_NULL

    as_buffer :: Ptr{CPyBufferProcs} = C_NULL

    flags :: Culong = 0

    doc :: Cstring = C_NULL

    traverse :: Ptr{Cvoid} = C_NULL

    clear :: Ptr{Cvoid} = C_NULL

    richcompare :: Ptr{Cvoid} = C_NULL

    weaklistoffset :: CPy_ssize_t = 0

    iter :: Ptr{Cvoid} = C_NULL
    iternext :: Ptr{Cvoid} = C_NULL

    methods :: Ptr{CPyMethodDefStruct} = C_NULL
    members :: Ptr{CPyMemberDefStruct} = C_NULL
    getset :: Ptr{CPyGetSetDefStruct} = C_NULL
    base :: CPyPtr = C_NULL
    dict :: CPyPtr = C_NULL
    descr_get :: Ptr{Cvoid} = C_NULL
    descr_set :: Ptr{Cvoid} = C_NULL
    dictoffset :: CPy_ssize_t = 0
    init :: Ptr{Cvoid} = C_NULL
    alloc :: Ptr{Cvoid} = C_NULL
    new :: Ptr{Cvoid} = C_NULL
    free :: Ptr{Cvoid} = C_NULL
    is_gc :: Ptr{Cvoid} = C_NULL
    bases :: CPyPtr = C_NULL
    mro :: CPyPtr = C_NULL
    cache :: CPyPtr = C_NULL
    subclasses :: CPyPtr = C_NULL
    weaklist :: CPyPtr = C_NULL
    del :: Ptr{Cvoid} = C_NULL

    version_tag :: Cuint = 0

    finalize :: Ptr{Cvoid} = C_NULL
    vectorcall :: Ptr{Cvoid} = C_NULL
end

const CPyTypePtr = Ptr{CPyTypeObject}

cpytypename(o::Ptr) = unsafe_string(unsafe_load(CPyTypePtr(o)).name)
cpytypeflags(o::Ptr) = unsafe_load(CPyTypePtr(o)).flags
cpytypemro(o::Ptr) = unsafe_load(CPyTypePtr(o)).mro
cpytypeissubtype(s::Ptr, t::Ptr) = cpycall_boolx(Val(:PyType_IsSubtype), s, t)
cpytypecheck(o::Ptr, t::Ptr) = cpytypeissubtype(cpytype(o), t)
cpytypecheckexact(o::Ptr, t::Ptr) = cpytype(o) == t
cpytypeissubtypefast(s::Ptr, f::Integer) = cpytypehasfeature(s, f)
cpytypehasfeature(s::Ptr, f::Integer) = !iszero(cpytypeflags(s) & f)
cpytypecheckfast(o::Ptr, f::Integer) = cpytypeissubtypefast(cpytype(o), f)

abstract type AbstractCPySimpleObject <: AbstractCPyObject end

@kwdef struct CPySimpleObject{T} <: AbstractCPySimpleObject
    ob_head :: CPyObject = CPyObject()
    value :: T
end

cpysimpleobjectvalue(::Type{T}, o::Ptr) where {T} = unsafe_load(Ptr{CPySimpleObject{T}}(o)).value
