PyFraction_Type() = begin
    ptr = POINTERS.PyFraction_Type
    if isnull(ptr)
        m = PyImport_ImportModule("fractions")
        isnull(m) && return ptr
        ptr = PyObject_GetAttrString(m, "Fraction")
        Py_DecRef(m)
        isnull(m) && return ptr
        POINTERS.PyFraction_Type = ptr
    end
    ptr
end

PyFraction_From(x::Union{Rational,Integer}) = begin
    t = PyFraction_Type()
    isnull(t) && return PyNULL
    a = PyTuple_New(2)
    isnull(a) && return PyNULL
    b = PyLong_From(numerator(x))
    isnull(b) && (Py_DecRef(a); return PyNULL)
    err = PyTuple_SetItem(a, 0, b)
    ism1(err) && (Py_DecRef(a); return PyNULL)
    b = PyLong_From(denominator(x))
    isnull(b) && (Py_DecRef(a); return PyNULL)
    err = PyTuple_SetItem(a, 1, b)
    ism1(err) && (Py_DecRef(a); return PyNULL)
    r = PyObject_CallObject(t, a)
    Py_DecRef(a)
    return r
end
