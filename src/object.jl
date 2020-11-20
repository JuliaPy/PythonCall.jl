abstract type AbstractPyObject end
export AbstractPyObject

pyincref!(o::AbstractPyObject) = (C.Py_IncRef(pyptr(o)); o)
Base.convert(::Type{AbstractPyObject}, o::AbstractPyObject) = o
Base.unsafe_convert(::Type{CPyPtr}, o::AbstractPyObject) = pyptr(o)

### PYOBJECT

mutable struct PyObject <: AbstractPyObject
    ptr :: CPyPtr
    function PyObject(::Val{:new}, ptr::Ptr=C_NULL, borrowed::Bool=false)
        o = new(CPyPtr(ptr))
        borrowed && C.Py_IncRef(ptr)
        finalizer(o) do o
            C.Py_DecRef(pyptr(o))
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

Base.convert(::Type{PyObject}, o::PyObject) = o
Base.convert(::Type{PyObject}, o) = PyObject(o)
Base.convert(::Type{PyObject}, o::AbstractPyObject) = PyObject(o)

Base.promote_rule(::Type{PyObject}, ::Type{<:AbstractPyObject}) = PyObject

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
pyobject(o::IO) = pyjl(o, BufferedIO{typeof(o)})
pyobject(o) = pyjl(o)
export pyobject

### ABSTRACT OBJECT API

pyrefcnt(o::AbstractPyObject) = C.Py_RefCnt(o)
export pyrefcnt

pyis(o1::AbstractPyObject, o2::AbstractPyObject) = pyptr(o1) == pyptr(o2)
export pyis

pyhasattr(o::AbstractPyObject, k::AbstractString) = check(Bool, C.PyObject_HasAttrString(o, k))
pyhasattr(o::AbstractPyObject, k::Symbol) = pyhasattr(o, String(k))
pyhasattr(o::AbstractPyObject, k) = check(Bool, C.PyObject_HasAttr(o, pyobject(k)))
export pyhasattr

pygetattr(o::AbstractPyObject, k::AbstractString) = check(C.PyObject_GetAttrString(o, k))
pygetattr(o::AbstractPyObject, k::Symbol) = pygetattr(o, String(k))
pygetattr(o::AbstractPyObject, k) = check(C.PyObject_GetAttr(o, pyobject(k)))
export pygetattr

pysetattr(o::AbstractPyObject, k::AbstractString, v) = check(Nothing, C.PyObject_SetAttrString(o, k, pyobject(v)))
pysetattr(o::AbstractPyObject, k::Symbol, v) = pysetattr(o, String(k), v)
pysetattr(o::AbstractPyObject, k, v) = check(Nothing, C.PyObject_SetAttr(o, pyobject(k), pyobject(v)))
export pysetattr

pydelattr(o::AbstractPyObject, k::AbstractString) = check(Nothing, C.PyObject_DelAttrString(o, k))
pydelattr(o::AbstractPyObject, k::Symbol) = pydelattr(o, String(k))
pydelattr(o::AbstractPyObject, k) = check(Nothing, C.PyObject_DelAttr(o, pyobject(k)))
export pydelattr

pyrichcompare(::Type{PyObject}, o1, o2, op::Cint) = check(C.PyObject_RichCompare(pyobject(o1), pyobject(o2), op))
pyrichcompare(::Type{Bool}, o1, o2, op::Cint) = check(Bool, C.PyObject_RichCompareBool(pyobject(o1), pyobject(o2), op))
pyrichcompare(o1, o2, op) = pyrichcompare(PyObject, o1, o2, op)
pyrichcompare(::Type{T}, o1, o2, ::typeof(< )) where {T} = pyrichcompare(T, o1, o2, C.Py_LT)
pyrichcompare(::Type{T}, o1, o2, ::typeof(<=)) where {T} = pyrichcompare(T, o1, o2, C.Py_LE)
pyrichcompare(::Type{T}, o1, o2, ::typeof(==)) where {T} = pyrichcompare(T, o1, o2, C.Py_EQ)
pyrichcompare(::Type{T}, o1, o2, ::typeof(!=)) where {T} = pyrichcompare(T, o1, o2, C.Py_NE)
pyrichcompare(::Type{T}, o1, o2, ::typeof(> )) where {T} = pyrichcompare(T, o1, o2, C.Py_GT)
pyrichcompare(::Type{T}, o1, o2, ::typeof(>=)) where {T} = pyrichcompare(T, o1, o2, C.Py_GE)
export pyrichcompare

pyeq(args...) = pyrichcompare(args..., ==)
pyne(args...) = pyrichcompare(args..., !=)
pylt(args...) = pyrichcompare(args..., < )
pyle(args...) = pyrichcompare(args..., <=)
pygt(args...) = pyrichcompare(args..., > )
pyge(args...) = pyrichcompare(args..., >=)
export pyeq, pyne, pylt, pyle, pygt, pyge

pyrepr(o) = check(C.PyObject_Repr(pyobject(o)))
pyrepr(::Type{String}, o) = pystr_asjuliastring(pyrepr(o))
export pyrepr

pyascii(o) = check(C.PyObject_ASCII(pyobject(o)))
pyascii(::Type{String}, o) = pystr_asjuliastring(pyascii(o))
export pyascii

pystr(o) = check(C.PyObject_Str(pyobject(o)))
pystr(::Type{String}, o) = pystr_asjuliastring(pystr(o))
export pystr

pybytes(o) = check(C.PyObject_Bytes(pyobject(o)))
export pybytes

pyissubclass(o1::AbstractPyObject, o2::AbstractPyObject) = check(Bool, C.PyObject_IsSubclass(o1, o2))
export pyissubclass

pyisinstance(o::AbstractPyObject, t::AbstractPyObject) = check(Bool, C.PyObject_IsInstance(o, t))
export pyisinstance

pyhash(o) = check(C.PyObject_Hash(pyobject(o)))
export pyhash

pytruth(o) = check(Bool, C.PyObject_IsTrue(pyobject(o)))
export pytruth

pylen(o) = check(C.PyObject_Length(pyobject(o)))
export pylen

pygetitem(o::AbstractPyObject, k) = check(C.PyObject_GetItem(o, pyobject(k)))
export pygetitem

pysetitem(o::AbstractPyObject, k, v) = check(Nothing, C.PyObject_SetItem(o, pyobject(k), pyobject(v)))
export pysetitem

pydelitem(o::AbstractPyObject, k) = check(Nothing, C.PyObject_DelItem(o, pyobject(k)))
export pydelitem

pydir(o::AbstractPyObject) = check(C.PyObject_Dir(o))
export pydir

pyiter(o) = check(C.PyObject_GetIter(pyobject(o)))
pyiter(args...; opts...) = pybuiltins.iter(args...; opts...)
export pyiter

pycall(f, args...; kwargs...) =
    if !isempty(kwargs)
        argso = pytuple_fromiter(args)
        kwargso = pydict_fromstringiter(kwargs)
        check(C.PyObject_Call(f, argso, kwargso))
    elseif !isempty(args)
        argso = pytuple_fromiter(args)
        check(C.PyObject_CallObject(f, argso))
    else
        check(C.PyObject_CallObject(f, C_NULL))
    end
export pycall

### MODULE OBJECTS

const pymoduletype = PyLazyObject(() -> pytype(pybuiltins))
export pymoduletype

pyismodule(o::AbstractPyObject) = pyisinstance(o, pymoduletype)
