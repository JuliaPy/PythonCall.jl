mutable struct PyObject
    ref :: PyRef
    make :: Any
    PyObject(::Val{:nocopy}, o::PyRef) = new(o)
    PyObject(o) = new(PyRef(o))
    PyObject(::Val{:lazy}, mk) = new(PyRef(), mk)
end
export PyObject

ispyreftype(::Type{PyObject}) = true
pyptr(o::PyObject) = begin
    ref = getfield(o, :ref)
    ptr = ref.ptr
    if isnull(ptr)
        val = try
            getfield(o, :make)()
        catch err
            C.PyErr_SetString(C.PyExc_Exception(), "Error retrieving object value: $err")
            return ptr
        end
        ptr = ref.ptr = C.PyObject_From(val)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, o::PyObject) = checknull(pyptr(o))
pynewobject(p::Ptr, check::Bool=false) = (check && isnull(p)) ? pythrow() : PyObject(Val(:nocopy), pynewref(p))
pyborrowedobject(p::Ptr, check::Bool=false) = (check && isnull(p)) ? pythrow() : PyObject(Val(:nocopy), pyborrowedref(p))
pylazyobject(mk) = PyObject(Val(:lazy), mk)

C.PyObject_TryConvert__initial(o, ::Type{PyObject}) = C.putresult(PyObject, pyborrowedobject(o))

Base.convert(::Type{PyObject}, x::PyObject) = x
Base.convert(::Type{PyObject}, x) = PyObject(x)

### Cache some common values

const _pynone = pylazyobject(() -> pynone(PyRef))
pynone(::Type{PyObject}) = _pynone

const _pytrue = pylazyobject(() -> pybool(PyRef, true))
const _pyfalse = pylazyobject(() -> pybool(PyRef, false))
pybool(::Type{PyObject}, x::Bool) = x ? _pytrue : _pyfalse

### IO

Base.string(o::PyObject) = pystr(String, o)

Base.print(io::IO, o::PyObject) = print(io, string(o))

Base.show(io::IO, o::PyObject) = begin
    s = pyrepr(String, o)
    if get(io, :typeinfo, Any) == typeof(o)
        print(io, s)
    elseif startswith(s, "<") && endswith(s, ">")
        print(io, "<py ", s[2:end])
    else
        print(io, "<py ", s, ">")
    end
end

### PROPERTIES

Base.getproperty(o::PyObject, k::Symbol) = pygetattr(PyObject, o, k)

Base.setproperty!(o::PyObject, k::Symbol, v) = pysetattr(o, k, v)

Base.hasproperty(o::PyObject, k::Symbol) = pyhasattr(o, k)

function Base.propertynames(o::PyObject)
    # this follows the logic of rlcompleter.py
    @py ```
    def members(o):
        def classmembers(c):
            r = dir(c)
            if hasattr(c, '__bases__'):
                for b in c.__bases__:
                    r += classmembers(b)
            return r
        words = set(dir(o))
        words.discard('__builtins__')
        if hasattr(o, '__class__'):
            words.add('__class__')
            words.update(classmembers(o.__class__))
        return words
    $words = members($o)
    ```
    [Symbol(pystr(String, x)) for x in words]
end

### CALL

(f::PyObject)(args...; kwargs...) = pycall(PyObject, f, args...; kwargs...)

### ITERATE & INDEX

Base.IteratorSize(::Type{PyObject}) = Base.SizeUnknown()
Base.eltype(::Type{PyObject}) = PyObject

Base.getindex(o::PyObject, k) = pygetitem(PyObject, o, k)
Base.getindex(o::PyObject, k...) = pygetitem(PyObject, o, k)
Base.setindex!(o::PyObject, v, k) = pysetitem(o, k, v)
Base.setindex!(o::PyObject, v, k...) = pysetitem(o, k, v)
Base.delete!(o::PyObject, k) = pydelitem(o, k)
Base.delete!(o::PyObject, k...) = pydelitem(o, k)

Base.length(o::PyObject) = Int(pylen(o))

Base.iterate(o::PyObject, it::PyRef) = begin
    vo = C.PyIter_Next(it)
    if !isnull(vo)
        (pynewobject(vo), it)
    elseif C.PyErr_IsSet()
        pythrow()
    else
        nothing
    end
end
Base.iterate(o::PyObject) = iterate(o, pynewref(C.PyObject_GetIter(o), true))

Base.in(x, o::PyObject) = pycontains(o, x)
Base.hash(o::PyObject) = trunc(UInt, pyhash(o))

### COMPARISON

Base.:(==)(x::PyObject, y::PyObject) = pyeq(PyObject, x, y)
Base.:(!=)(x::PyObject, y::PyObject) = pynq(PyObject, x, y)
Base.:(<=)(x::PyObject, y::PyObject) = pyle(PyObject, x, y)
Base.:(< )(x::PyObject, y::PyObject) = pylt(PyObject, x, y)
Base.:(>=)(x::PyObject, y::PyObject) = pyge(PyObject, x, y)
Base.:(> )(x::PyObject, y::PyObject) = pygt(PyObject, x, y)
Base.isequal(x::PyObject, y::PyObject) = pyeq(Bool, x, y)
Base.isless(x::PyObject, y::PyObject) = pylt(Bool, x, y)
