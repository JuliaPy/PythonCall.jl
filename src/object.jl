### PYOBJECT

mutable struct PyObject
    ptr :: CPyPtr
    get :: Function
    function PyObject(::Val{:new}, src::Union{Function,Ptr}, borrowed::Bool=false)
        if src isa Ptr
            o = new(CPyPtr(src))
            borrowed && C.Py_IncRef(src)
        else
            o = new(CPyPtr(C_NULL), src)
        end
        finalizer(o) do o
            if CONFIG.isinitialized
                ptr = getfield(o, :ptr)
                if ptr != C_NULL
                    with_gil() do
                        C.Py_DecRef(ptr)
                    end
                end
            end
            setfield!(o, :ptr, CPyPtr(C_NULL))
            nothing
        end
        o
    end
end
export PyObject

"""
    pynewobject(ptr::Ptr)

Construct a `PyObject` from the new reference `ptr`.
"""
pynewobject(ptr::Ptr) = PyObject(Val(:new), ptr, false)

"""
    pyborrowedobject(ptr::Ptr)

Construct a `PyObject` from the borrowed reference `ptr`. Its refcount is incremented.
"""
pyborrowedobject(ptr::Ptr) = PyObject(Val(:new), ptr, true)

"""
    pylazyobject(f::Function)

Construct a `PyObject` lazily whose value is equal to `f()`, except evaluation of `f` is deferred until the object is used at run-time.

Use these to construct module-level constants, such as `pynone`.
"""
pylazyobject(f::Function) = PyObject(Val(:new), f)

function pyptr(o::PyObject)
    # check if assigned
    ptr = getfield(o, :ptr)
    ptr == C_NULL || return ptr
    # otherwise generate a value
    val = getfield(o, :get)()
    ptr = pyptr(val)
    setfield!(o, :ptr, ptr)
    C.Py_IncRef(ptr)
    return ptr
end

PyObject(o::PyObject) = o
PyObject(o) = PyObject(pyobject(o))

Base.convert(::Type{PyObject}, o::PyObject) = o
Base.convert(::Type{PyObject}, o) = PyObject(o)

pyincref!(o::PyObject) = (C.Py_IncRef(pyptr(o)); o)

Base.unsafe_convert(::Type{CPyPtr}, o::PyObject) = pyptr(o)

### CONVERSIONS

const pyobjecttype = pylazyobject(() -> pybuiltins.object)
export pyobjecttype

pyobject(o::PyObject) = o
pyobject(args...; opts...) = pyobjecttype(args...; opts...)
pyobject(o::Nothing) = pynone
pyobject(o::Missing) = pynone
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
pyobject(o::IO) = pybufferedio(o)
pyobject(o) = pyjl(o)
export pyobject

### ABSTRACT OBJECT API

pyrefcnt(o::PyObject) = C.Py_RefCnt(o)
export pyrefcnt

pyis(o1::PyObject, o2::PyObject) = pyptr(o1) == pyptr(o2)
export pyis

pyhasattr(o::PyObject, k::AbstractString) = check(Bool, C.PyObject_HasAttrString(o, k))
pyhasattr(o::PyObject, k::Symbol) = pyhasattr(o, String(k))
pyhasattr(o::PyObject, k) = check(Bool, C.PyObject_HasAttr(o, pyobject(k)))
export pyhasattr

pygetattr(o::PyObject, k::AbstractString) = check(C.PyObject_GetAttrString(o, k))
pygetattr(o::PyObject, k::Symbol) = pygetattr(o, String(k))
pygetattr(o::PyObject, k) = check(C.PyObject_GetAttr(o, pyobject(k)))
export pygetattr

pysetattr(o::PyObject, k::AbstractString, v) = check(Nothing, C.PyObject_SetAttrString(o, k, pyobject(v)))
pysetattr(o::PyObject, k::Symbol, v) = pysetattr(o, String(k), v)
pysetattr(o::PyObject, k, v) = check(Nothing, C.PyObject_SetAttr(o, pyobject(k), pyobject(v)))
export pysetattr

pydelattr(o::PyObject, k::AbstractString) = check(Nothing, C.PyObject_DelAttrString(o, k))
pydelattr(o::PyObject, k::Symbol) = pydelattr(o, String(k))
pydelattr(o::PyObject, k) = check(Nothing, C.PyObject_DelAttr(o, pyobject(k)))
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

pyissubclass(o1::PyObject, o2::PyObject) = check(Bool, C.PyObject_IsSubclass(o1, o2))
export pyissubclass

pyisinstance(o::PyObject, t::PyObject) = check(Bool, C.PyObject_IsInstance(o, t))
export pyisinstance

pyhash(o) = check(C.PyObject_Hash(pyobject(o)))
export pyhash

pytruth(o) = check(Bool, C.PyObject_IsTrue(pyobject(o)))
export pytruth

pylen(o) = check(C.PyObject_Length(pyobject(o)))
export pylen

pygetitem(o::PyObject, k) = check(C.PyObject_GetItem(o, pyobject(k)))
export pygetitem

pysetitem(o::PyObject, k, v) = check(Nothing, C.PyObject_SetItem(o, pyobject(k), pyobject(v)))
export pysetitem

pydelitem(o::PyObject, k) = check(Nothing, C.PyObject_DelItem(o, pyobject(k)))
export pydelitem

pydir(o::PyObject) = check(C.PyObject_Dir(o))
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

const pymoduletype = pylazyobject(() -> pytype(pybuiltins))
export pymoduletype

pyismodule(o::PyObject) = pyisinstance(o, pymoduletype)
