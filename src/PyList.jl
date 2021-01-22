"""
    PyList{T=PyObject}([o])

Wrap the Python list `o` (or anything satisfying the sequence interface) as a Julia vector with elements of type `T`.

If `o` is not given, an empty list is created.
"""
struct PyList{T} <: AbstractVector{T}
    ref :: PyRef
    PyList{T}(o) where {T} = new{T}(ispyref(o) ? PyRef(o) : pylist(PyRef, o))
    PyList{T}() where {T} = new{T}(PyRef())
end
PyList(o) = PyList{PyObject}(o)
PyList() = PyList{PyObject}()
export PyList

ispyreftype(::Type{<:PyList}) = true
pyptr(x::PyList) = begin
    ptr = x.ref.ptr
    if isnull(ptr)
        ptr = x.ref.ptr = C.PyList_New(0)
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyList) = checknull(pyptr(x))
C.PyObject_TryConvert__initial(o, ::Type{T}) where {T<:PyList} = C.putresult(T(pyborrowedref(o)))

# Base.length(x::PyList) = @pyv `len($x)`::Int
Base.length(x::PyList) = Int(pylen(x))

Base.size(x::PyList) = (length(x),)

Base.getindex(x::PyList{T}, i::Integer) where {T} = begin
    checkbounds(x, i)
    # The following line implements this function, but is typically 3x slower.
    # @pyv `$x[$(i-1)]`::T
    p = checknull(C.PySequence_GetItem(x, i-1))
    r = C.PyObject_Convert(p, T)
    C.Py_DecRef(p)
    checkm1(r)
    C.takeresult(T)
end

Base.setindex!(x::PyList{T}, _v, i::Integer) where {T} = begin
    checkbounds(x, i)
    v = convertref(T, _v)
    # The following line implements this function, but is typically 10x slower
    # @py `$x[$(i-1)] = $v`
    vp = checknull(C.PyObject_From(v))
    err = C.PySequence_SetItem(x, i-1, vp)
    C.Py_DecRef(vp)
    checkm1(err)
    x
end

Base.insert!(x::PyList{T}, i::Integer, v) where {T} = (i==length(x)+1 || checkbounds(x, i); @py `$x.insert($(i-1), $(convertref(T, v)))`; x)

Base.push!(x::PyList{T}, v) where {T} = (@py `$x.append($(convertref(T, v)))`; x)

Base.pushfirst!(x::PyList, v) = insert!(x, 1, v)

Base.pop!(x::PyList{T}) where {T} = @pyv `$x.pop()`::T

Base.popat!(x::PyList{T}, i::Integer) where {T} = (checkbounds(x, i); @pyv `$x.pop($(i-1))`::T)

Base.popfirst!(x::PyList) = pop!(x, 1)

Base.reverse!(x::PyList) = (@py `$x.reverse()`; x)

# TODO: support kwarg `by` (becomes python kwarg `key`)
# Base.sort!(x::PyList; rev::Bool=false) = (@py `$x.sort(reverse=$rev)`; x)

Base.empty!(x::PyList) = (@py `$x.clear()`; x)

Base.copy(x::PyList) = @pyv `$x.copy()`::typeof(x)
