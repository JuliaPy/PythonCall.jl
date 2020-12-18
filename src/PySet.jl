"""
    PySet{T=PyObject}([o])

Wrap the Python set `o` (or anything satisfying the set interface) as a Julia set with elements of type `T`.

If `o` is not given, an empty set is created.
"""
struct PySet{T} <: AbstractSet{T}
    ref :: PyRef
    PySet{T}(o) where {T} = new{T}(PyRef(o))
    PySet{T}() where {T} = new{T}(PyRef())
end
PySet(o) = PySet{PyObject}(o)
PySet() = PySet{PyObject}()
export PySet

ispyreftype(::Type{<:PySet}) = true
pyptr(x::PySet) = begin
    ptr = x.ref.ptr
    if isnull(ptr)
        ptr = x.ref.ptr = C.PySet_New(C_NULL)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PySet) = checknull(pyptr(x))
C.PyObject_TryConvert__initial(o, ::Type{T}) where {T<:PySet} = C.putresult(T, T(pyborrowedref(o)))

Base.iterate(x::PySet{T}, it::PyRef) where {T} = begin
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
Base.iterate(x::PySet) = begin
    it = C.PyObject_GetIter(x)
    isnull(it) && pythrow()
    iterate(x, pynewref(it))
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
