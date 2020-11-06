abstract type AbstractPyObject end
export AbstractPyObject

pyincref!(o::AbstractPyObject) = (cpyincref(pyptr(o)); o)

### PYOBJECT

mutable struct PyObject <: AbstractPyObject
    ptr :: CPyPtr
    function PyObject(::Val{:new}, ptr::Ptr=C_NULL, borrowed::Bool=false)
        o = new(CPyPtr(ptr))
        borrowed && cpyincref(ptr)
        finalizer(o) do o
            # pyptr(o)==C_NULL || @debug "decref" ptr=pyptr(o) refcnt=cpyrefcnt(pyptr(o))
            cpydecref(pyptr(o))
            setfield!(o, :ptr, CPyPtr(C_NULL))
            nothing
        end
    end
end
PyObject(o::PyObject) = o
PyObject(o::AbstractPyObject) = pynewobject(pyptr(o), true)
PyObject(o) = PyObject(pyobject(o))
export PyObject

pynewobject(args...) = PyObject(Val(:new), args...)

pyptr(o::PyObject) = getfield(o, :ptr)

### PYLAZYOBJECT

struct PyLazyObject{F} <: AbstractPyObject
    func :: F
    value :: PyObject
end
PyLazyObject(f) = PyLazyObject(f, pynewobject())

function pyptr(o::PyLazyObject)
    # check if assigned
    val = getfield(o, :value)
    ptr = pyptr(val)
    ptr == C_NULL || return ptr
    # otherwise, generate a value
    newval = getfield(o, :func)()
    ptr = pyptr(newval)
    setfield!(val, :ptr, ptr)
    return ptr
end

PyObject(o::PyLazyObject) = (pyptr(o); getfield(o, :value))

### CONVERSIONS

const pyobjecttype = PyLazyObject(() -> pybuiltins.object)
export pyobjecttype

pyobject(o::AbstractPyObject) = o
pyobject(args...; opts...) = pyobjecttype(args...; opts...)
pyobject(o::Nothing) = pynone
pyobject(o::Tuple) = pytuple(o)
pyobject(o::Pair) = pytuple(o)
pyobject(o::AbstractString) = pystr(o)
pyobject(o::Bool) = pybool(o)
pyobject(o::Integer) = pyint(o)
pyobject(o::Union{Float16,Float32,Float64}) = pyfloat(o)
pyobject(o::Union{Complex{Float16}, Complex{Float32}, Complex{Float64}}) = pycomplex(o)
pyobject(o::Rational) = pyfraction(o)
pyobject(o::AbstractRange{<:Integer}) = pyrange(o)
pyobject(o::DateTime) = pydatetime(o)
pyobject(o::Date) = pydate(o)
pyobject(o::Time) = pytime(o)
pyobject(o) = pyjulia(o)
export pyobject

Base.convert(::Type{AbstractPyObject}, o::AbstractPyObject) = o
Base.convert(::Type{PyObject}, o::PyObject) = o
Base.convert(::Type{PyObject}, o) = PyObject(o)

### ABSTRACT OBJECT API

pyrefcnt(o::AbstractPyObject) = cpyrefcnt(pyptr(o))
export pyrefcnt

pyis(o1::AbstractPyObject, o2::AbstractPyObject) = pyptr(o1) == pyptr(o2)
export pyis

pyhasattr(o::AbstractPyObject, k::AbstractString) = cpycall_bool(Val(:PyObject_HasAttrString), o, k)
pyhasattr(o::AbstractPyObject, k::Symbol) = pyhasattr(o, String(k))
pyhasattr(o::AbstractPyObject, k) = cpycall_bool(Val(:PyObject_HasAttr), o, pyobject(k))
export pyhasattr

pygetattr(o::AbstractPyObject, k::AbstractString) = cpycall_obj(Val(:PyObject_GetAttrString), o, k)
pygetattr(o::AbstractPyObject, k::Symbol) = pygetattr(o, String(k))
pygetattr(o::AbstractPyObject, k) = cpycall_obj(Val(:PyObject_GetAttr), o, pyobject(k))
export pygetattr

pysetattr(o::AbstractPyObject, k::AbstractString, v) = cpycall_void(Val(:PyObject_SetAttrString), o, k, pyobject(v))
pysetattr(o::AbstractPyObject, k::Symbol, v) = pysetattr(o, String(k), v)
pysetattr(o::AbstractPyObject, k, v) = cpycall_void(Val(:PyObject_SetAttr), o, pyobject(k), pyobject(v))
export pysetattr

pydelattr(o::AbstractPyObject, k::AbstractString) = cpycall_void(Val(:PyObject_DelAttrString), o, k)
pydelattr(o::AbstractPyObject, k::Symbol) = pydelattr(o, String(k))
pydelattr(o::AbstractPyObject, k) = cpycall_void(Val(:PyObject_DelAttr), o, k)
export pydelattr

pyrichcompare(::Type{PyObject}, o1, o2, op::CPy_CompareOp) = cpycall_obj(Val(:PyObject_RichCompare), pyobject(o1), pyobject(o2), op)
pyrichcompare(::Type{Bool}, o1, o2, op::CPy_CompareOp) = cpycall_bool(Val(:PyObject_RichCompareBool), pyobject(o1), pyobject(o2), op)
pyrichcompare(o1, o2, op) = pyrichcompare(PyObject, o1, o2, op)
pyrichcompare(::Type{T}, o1, o2, ::typeof(< )) where {T} = pyrichcompare(T, o1, o2, CPy_LT)
pyrichcompare(::Type{T}, o1, o2, ::typeof(<=)) where {T} = pyrichcompare(T, o1, o2, CPy_LE)
pyrichcompare(::Type{T}, o1, o2, ::typeof(==)) where {T} = pyrichcompare(T, o1, o2, CPy_EQ)
pyrichcompare(::Type{T}, o1, o2, ::typeof(!=)) where {T} = pyrichcompare(T, o1, o2, CPy_NE)
pyrichcompare(::Type{T}, o1, o2, ::typeof(> )) where {T} = pyrichcompare(T, o1, o2, CPy_GT)
pyrichcompare(::Type{T}, o1, o2, ::typeof(>=)) where {T} = pyrichcompare(T, o1, o2, CPy_GE)
export pyrichcompare

pyeq(args...) = pyrichcompare(args..., ==)
pyne(args...) = pyrichcompare(args..., !=)
pylt(args...) = pyrichcompare(args..., < )
pyle(args...) = pyrichcompare(args..., <=)
pygt(args...) = pyrichcompare(args..., > )
pyge(args...) = pyrichcompare(args..., >=)
export pyeq, pyne, pylt, pyle, pygt, pyge

pyrepr(o) = cpycall_obj(Val(:PyObject_Repr), pyobject(o))
pyrepr(::Type{String}, o) = pystr_asjuliastring(pyrepr(o))
export pyrepr

pyascii(o) = cpycall_obj(Val(:PyObject_ASCII), pyobject(o))
pyascii(::Type{String}, o) = pystr_asjuliastring(pyascii(o))
export pyascii

pystr(o) = cpycall_obj(Val(:PyObject_Str), pyobject(o))
pystr(::Type{String}, o) = pystr_asjuliastring(pystr(o))
export pystr

pybytes(o) = cpycall_obj(Val(:PyObject_Bytes), pyobject(o))
export pybytes

pyissubclass(o1::AbstractPyObject, o2::AbstractPyObject) = cpycall_bool(Val(:PyObject_IsSubclass), o1, o2)
export pyissubclass

pyisinstance(o::AbstractPyObject, t::AbstractPyObject) = cpycall_bool(Val(:PyObject_IsInstance), o, t)
export pyisinstance

pyhash(o) = cpycall_num(Val(:PyObject_Hash), CPy_hash_t, pyobject(o))
export pyhash

pytruth(o) = cpycall_bool(Val(:PyObject_IsTrue), pyobject(o))
export pytruth

pylen(o) = cpycall_num(Val(:PyObject_Length), CPy_ssize_t, pyobject(o))
export pylen

pygetitem(o::AbstractPyObject, k) = cpycall_obj(Val(:PyObject_GetItem), o, pyobject(k))
export pygetitem

pysetitem(o::AbstractPyObject, k, v) = cpycall_void(Val(:PyObject_SetItem), o, pyobject(k), pyobject(v))
export pysetitem

pydelitem(o::AbstractPyObject, k) = cpycall_void(Val(:PyObject_DelItem), o, pyobject(k))
export pydelitem

pydir(o::AbstractPyObject) = cpycall_obj(Val(:PyObject_Dir), o)
export pydir

pyiter(o) = cpycall_obj(Val(:PyObject_GetIter), o)
export pyiter

pycall(f, args...; kwargs...) =
    if !isempty(kwargs)
        argso = pytuple_fromiter(args)
        kwargso = pydict_fromstringiter(kwargs)
        cpycall_obj(Val(:PyObject_Call), f, argso, kwargso)
    elseif !isempty(args)
        argso = pytuple_fromiter(args)
        cpycall_obj(Val(:PyObject_CallObject), f, argso)
    else
        cpycall_obj(Val(:PyObject_CallObject), f, C_NULL)
    end
export pycall
