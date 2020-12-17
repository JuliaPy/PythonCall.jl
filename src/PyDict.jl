"""
    PyDict{K=PyObject, V=PyObject}([o])

Wrap the Python dictionary `o` (or anything satisfying the mapping interface) as a Julia dictionary with keys of type `K` and values of type `V`.

If `o` is not given, an empty dict is created.
"""
mutable struct PyDict{K,V} <: AbstractDict{K,V}
    ref :: PyRef
    hasbuiltins :: Bool
    PyDict{K,V}(o) where {K,V} = new(PyRef(o), false)
    PyDict{K,V}() where {K,V} = new(PyRef(), false)
end
PyDict{K}(args...) where {K} = PyDict{K,PyObject}(args...)
PyDict(args...) = PyDict{PyObject,PyObject}(args...)
export PyDict

const pyglobals = PyDict{String}()
export pyglobals

ispyreftype(::Type{<:PyDict}) = true
pyptr(x::PyDict) = begin
    ptr = x.ref.ptr
    if isnull(ptr)
        ptr = x.ref.ptr = C.PyDict_New()
    end
    ptr
end
Base.unsafe_convert(::Type{CPyPtr}, x::PyDict) = checknull(pyptr(x))

Base.iterate(x::PyDict{K,V}, it::PyRef) where {K,V} = begin
    ptr = C.PyIter_Next(it)
    if !isnull(ptr)
        ko = C.PySequence_GetItem(ptr, 0)
        isnull(ko) && pythrow()
        r = C.PyObject_Convert(ko, K)
        C.Py_DecRef(ko)
        ism1(r) && pythrow()
        k = C.takeresult(K)
        vo = C.PySequence_GetItem(ptr, 1)
        isnull(vo) && pythrow()
        r = C.PyObject_Convert(vo, V)
        C.Py_DecRef(vo)
        ism1(r) && pythrow()
        v = C.takeresult(V)
        (k => v, it)
    elseif C.PyErr_IsSet()
        pythrow()
    else
        nothing
    end
end
Base.iterate(x::PyDict) = begin
    a = C.PyObject_GetAttrString(x, "items")
    isnull(a) && pythrow()
    b = C.PyObject_CallNice(a)
    C.Py_DecRef(a)
    isnull(b) && pythrow()
    it = C.PyObject_GetIter(b)
    C.Py_DecRef(b)
    isnull(it) && pythrow()
    iterate(x, pynewref(it))
end

Base.setindex!(x::PyDict{K,V}, v, k) where {K,V} = pysetitem(x, convertref(K, k), convertref(V, v))

Base.getindex(x::PyDict{K,V}, k) where {K,V} = pygetitem(V, x, convertref(K, k))

Base.delete!(x::PyDict{K}, k) where {K} = pydelitem(x, convertref(K, k))

Base.length(x::PyDict) = Int(pylen(x))

Base.empty!(x::PyDict) = (@py `$x.clear()`; x)

Base.copy(x::PyDict) = @pyv typeof(x) `$x.copy()`

Base.haskey(x::PyDict{K}, _k) where {K} = begin
    k = tryconvertref(K, _k)
    k === PYERR() && pythrow()
    k === NOTIMPLEMENTED() && return false
    pycontains(x, k)
end

Base.get(x::PyDict{K}, _k, d) where {K} = begin
    k = tryconvertref(K, _k)
    k === PYERR() && pythrow()
    (k !== NOTIMPLEMENTED() && pycontains(x, k)) ? x[k] : d
end

Base.get(d::Function, x::PyDict{K}, _k) where {K} = begin
    k = tryconvertref(K, _k)
    k === PYERR() && pythrow()
    (k !== NOTIMPLEMENTED() && pycontains(x, k)) ? x[k] : d()
end

Base.get!(x::PyDict{K,V}, _k, d) where {K,V} = begin
    k = convertref(K, _k)
    pycontains(x, k) ? x[k] : (x[k] = convert(V, d))
end

Base.get!(d::Function, x::PyDict{K,V}, _k) where {K,V} = begin
    k = convertref(K, _k)
    pycontains(x, k) ? x[k] : (x[k] = convert(V, d()))
end
