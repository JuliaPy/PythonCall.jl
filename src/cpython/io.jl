for n in [:IOBase, :RawIOBase, :BufferedIOBase, :TextIOBase]
    p = Symbol(:Py, n)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyPtr())
    @eval $t(doimport::Bool=true) = begin
        ptr = $tr[]
        isnull(ptr) || return ptr
        a = doimport ? PyImport_ImportModule("io") : PyImport_GetModule("io")
        isnull(a) && return a
        b = PyObject_GetAttrString(a, $(string(n)))
        Py_DecRef(a)
        isnull(b) && return b
        $tr[] = b
    end
    @eval $c(o) = begin
        t = $t(false)
        isnull(t) && return (PyErr_IsSet() ? Cint(-1) : Cint(0))
        PyObject_IsInstance(o, t)
    end
end
