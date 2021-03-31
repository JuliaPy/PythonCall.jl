"""
    PySet{T=PyObject}([o])

Wrap the Python set `o` (or anything satisfying the set interface) as a Julia set with elements of type `T`.

If `o` is not a Python object, it must be an iterable and is converted to a Python set.

If `o` is not given, an empty set is created.
"""
mutable struct PySet{T} <: AbstractSet{T}
    ptr::CPyPtr
    PySet{T}(::Val{:new}, ptr::Ptr) where {T} = finalizer(pyref_finalize!, new{T}(CPyPtr(ptr)))
end
PySet{T}(o) where {T} = PySet{T}(Val(:new), checknull(ispyref(o) ? C.PyObject_From(o) : C.PySet_FromIter(o)))
PySet{T}() where {T} = PySet{T}(Val(:new), CPyPtr(0))
PySet(o) = PySet{PyObject}(o)
PySet() = PySet{PyObject}()
export PySet

ispyreftype(::Type{<:PySet}) = true
pyptr(x::PySet) = begin
    ptr = x.ptr
    if isnull(ptr)
        ptr = x.ptr = C.PySet_New(C_NULL)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PySet) = checknull(pyptr(x))
C.PyObject_TryConvert__initial(o, ::Type{PySet}) =
    C.PyObject_TryConvert__initial(o, PySet{PyObject})
C.PyObject_TryConvert__initial(o, ::Type{PySet{T}}) where {T} = begin
    C.Py_IncRef(o)
    C.putresult(PySet{T}(Val(:new), o))
end

Base.iterate(x::PySet{T}, it::PyRef = pyiter(PyRef, x)) where {T} = begin
    ptr = C.PyIter_Next(it)
    if ptr != C_NULL
        r = C.PyObject_Convert(ptr, T)
        C.Py_DecRef(ptr)
        checkm1(r)
        (C.takeresult(T), it)
    elseif C.PyErr_IsSet()
        pythrow()
    else
        nothing
    end
end

# Base.length(x::PySet) = @pyv `len($x)`::Int
Base.length(x::PySet) = Int(checkm1(C.PyObject_Length(x)))

Base.in(_v, x::PySet{T}) where {T} = begin
    v = tryconvertref(T, _v)
    v === PYERR() && pythrow()
    v === NOTIMPLEMENTED() && return false
    pycontains(x, v)
end

Base.push!(x::PySet{T}, v) where {T} = (@py `$x.add($(convertref(T, v)))`; x)

Base.delete!(x::PySet{T}, _v) where {T} = begin
    v = tryconvertref(T, _v)
    v === PYERR() && pythrow()
    v === NOTIMPLEMENTED() && return x
    @py `$x.discard($v)`
    x
end

Base.pop!(x::PySet{T}) where {T} = @pyv `$x.pop()`::T

Base.pop!(x::PySet{T}, _v) where {T} = begin
    v = tryconvert(T, _v)
    v === PYERR() && pythrow()
    (v !== NOTIMPLEMENTED() && v in x) ? (delete!(x, v); v) : error("not an element")
end

Base.pop!(x::PySet{T}, _v, d) where {T} = begin
    v = tryconvert(T, _v)
    v === PYERR() && pythrow()
    (v !== NOTIMPLEMENTED() && v in x) ? (delete!(x, v); v) : d
end

Base.empty!(x::PySet) = (@py `$x.clear()`; x)

Base.copy(x::PySet) = @pyv `$x.copy()`::typeof(x)
