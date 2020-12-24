const PyFraction_Type__ref = Ref(PyPtr())
PyFraction_Type() = begin
    ptr = PyFraction_Type__ref[]
    if isnull(ptr)
        m = PyImport_ImportModule("fractions")
        isnull(m) && return ptr
        ptr = PyObject_GetAttrString(m, "Fraction")
        Py_DecRef(m)
        isnull(m) && return ptr
        PyFraction_Type__ref[] = ptr
    end
    ptr
end

PyFraction_From(x::Union{Rational,Integer}) = begin
    t = PyFraction_Type()
    isnull(t) && return PyPtr()
    a = PyTuple_New(2)
    isnull(a) && return PyPtr()
    b = PyLong_From(numerator(x))
    isnull(b) && (Py_DecRef(a); return PyPtr())
    err = PyTuple_SetItem(a, 0, b)
    ism1(err) && (Py_DecRef(a); return PyPtr())
    b = PyLong_From(denominator(x))
    isnull(b) && (Py_DecRef(a); return PyPtr())
    err = PyTuple_SetItem(a, 1, b)
    ism1(err) && (Py_DecRef(a); return PyPtr())
    r = PyObject_CallObject(t, a)
    Py_DecRef(a)
    return r
end
