const PyJuliaSetValue_Type__ref = Ref(PyPtr())
PyJuliaSetValue_Type() = begin
    ptr = PyJuliaSetValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.SetValue",
            base = base,
            methods = [
                (name="add", flags=Py_METH_O, meth=pyjlset_add),
                (name="clear", flags=Py_METH_NOARGS, meth=pyjlset_clear),
                (name="copy", flags=Py_METH_NOARGS, meth=pyjlset_copy),
                (name="difference", flags=Py_METH_O, meth=pyjlset_difference),
                (name="difference_update", flags=Py_METH_O, meth=pyjlset_difference_update),
                (name="discard", flags=Py_METH_O, meth=pyjlset_discard),
                (name="intersection", flags=Py_METH_O, meth=pyjlset_intersection),
                (name="intersection_update", flags=Py_METH_O, meth=pyjlset_intersection_update),
                (name="isdisjoint", flags=Py_METH_O, meth=pyjlset_isdisjoint),
                # (name="issubset", flags=Py_METH_O, meth=pyjlset_issubset),
                # (name="issuperset", flags=Py_METH_O, meth=pyjlset_issuperset),
                (name="pop", flags=Py_METH_NOARGS, meth=pyjlset_pop),
                (name="remove", flags=Py_METH_O, meth=pyjlset_remove),
                (name="symmetric_difference", flags=Py_METH_O, meth=pyjlset_symmetric_difference),
                (name="symmetric_difference_update", flags=Py_METH_O, meth=pyjlset_symmetric_difference_update),
                (name="union", flags=Py_METH_O, meth=pyjlset_union),
                (name="update", flags=Py_METH_O, meth=pyjlset_update),
            ],
            # as_number = (
            #     or = pyjlset_or,
            #     and = pyjlset_and,
            #     xor = pyjlset_xor,
            #     subtract = pyjlset_sub,
            #     inplace_or = pyjlset_ior,
            #     inplace_and = pyjlset_iand,
            #     inplace_xor = pyjlset_ixor,
            #     inplace_subtract = pyjlset_isub,
            # ),
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyMutableSetABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaSetValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaSetValue_New(x::AbstractSet) = PyJuliaValue_New(PyJuliaSetValue_Type(), x)
PyJuliaValue_From(x::AbstractSet) = PyJuliaSetValue_New(x)

pyjlset_add(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    ism1(PyObject_Convert(vo, eltype(x))) && return PyPtr()
    v = takeresult(eltype(x))
    try
        push!(x, v)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_discard(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyPtr()
    r ==  0 && return PyNone_New()
    v = takeresult(eltype(x))
    try
        delete!(x, v)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_clear(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    try
        empty!(x)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_copy(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    try
        PyObject_From(copy(x))
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_pop(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    try
        if isempty(x)
            PyErr_SetString(PyExc_KeyError(), "pop from an empty set")
            PyPtr()
        else
            PyObject_From(pop!(x))
        end
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_remove(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    r = PyObject_TryConvert(vo, eltype(x))
    if r == -1
        return PyPtr()
    elseif r == 0
        PyErr_SetObject(PyExc_KeyError(), vo)
        return PyPtr()
    end
    v = takeresult(eltype(x))
    try
        if v in x
            delete!(x, v)
            PyNone_New()
        else
            PyErr_SetObject(PyExc_KeyError(), vo)
            PyPtr()
        end
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end

pyjlset_ibinop_named(xo, yo, op, skip) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    y = PyIterable_Collect(yo, eltype(x), skip)
    isempty(y) && PyErr_IsSet() && return PyPtr()
    try
        op(x, y)
        PyNone_New()
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end
pyjlset_update(xo, yo) = pyjlset_ibinop_named(xo, yo, union!, false)
pyjlset_difference_update(xo, yo) = pyjlset_ibinop_named(xo, yo, setdiff!, true)
pyjlset_symmetric_difference_update(xo, yo) = pyjlset_ibinop_named(xo, yo, symdiff!, false)
pyjlset_intersection_update(xo, yo) = pyjlset_ibinop_named(xo, yo, intersect!, true)

# pyjlset_ibinop_operator(xo, yo, op, skip) = begin
#     r = PySetABC_Check(yo)
#     r == -1 && return PyPtr()
#     r ==  0 && return PyNotImplemented_New()
#     x = PyJuliaValue_GetValue(xo)::AbstractSet
#     y = PyIterable_Collect(yo, eltype(x), skip)
#     isempty(y) && PyErr_IsSet() && return PyPtr()
#     try
#         PyObject_From(op(x, y))
#     catch err
#         PyErr_SetJuliaError(err)
#         PyPtr()
#     end
# end
# pyjlset_ior(xo, yo) = pyjlset_ibinop_operator(xo, yo, union!, false)
# pyjlset_isub(xo, yo) = pyjlset_ibinop_operator(xo, yo, setdiff!, true)
# pyjlset_ixor(xo, yo) = pyjlset_ibinop_operator(xo, yo, symdiff!, false)
# pyjlset_iand(xo, yo) = pyjlset_ibinop_operator(xo, yo, intersect!, true)

pyjlset_binop_generic(xo::PyPtr, yo::PyPtr, op) = begin
    xo2 = PySet_New(xo)
    isnull(xo2) && return PyPtr()
    yo2 = PySet_New(yo)
    isnull(yo2) && (Py_DecRef(xo2); return PyPtr())
    ro = op(xo2, yo2)
    Py_DecRef(xo2)
    Py_DecRef(yo2)
    ro
end
pyjlset_or_generic(xo, yo) = pyjlset_binop_generic(xo, yo, PyNumber_InPlaceOr)
pyjlset_xor_generic(xo, yo) = pyjlset_binop_generic(xo, yo, PyNumber_InPlaceXor)
pyjlset_and_generic(xo, yo) = pyjlset_binop_generic(xo, yo, PyNumber_InPlaceAnd)
pyjlset_sub_generic(xo, yo) = pyjlset_binop_generic(xo, yo, PyNumber_InPlaceSubtract)

pyjlset_binop_special(xo::PyPtr, yo::PyPtr, op) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    y = PyJuliaValue_GetValue(yo)::AbstractSet
    try
        PyObject_From(op(x, y))
    catch err
        PyErr_SetJuliaError(err)
        PyPtr()
    end
end
pyjlset_or_special(xo, yo) = pyjlset_binop_special(xo, yo, union)
pyjlset_xor_special(xo, yo) = pyjlset_binop_special(xo, yo, symdiff)
pyjlset_and_special(xo, yo) = pyjlset_binop_special(xo, yo, intersect)
pyjlset_sub_special(xo, yo) = pyjlset_binop_special(xo, yo, setdiff)

# pyjlset_binop_operator(xo::PyPtr, yo::PyPtr, gop, sop) = begin
#     t = PyJuliaSetValue_Type()
#     isnull(t) && return PyPtr()
#     if (r=PyObject_IsInstance(xo, t); ism1(r) && return PyPtr(); r!=0)
#         if (r=PyObject_IsInstance(yo, t); ism1(r) && return PyPtr(); r!=0)
#             sop(xo, yo)
#         elseif (r=PySetABC_Check(yo); ism1(r) && return PyPtr(); r!=0)
#             gop(xo, yo)
#         else
#             PyNotImplemented_New()
#         end
#     elseif (r=PyObject_IsInstance(yo, t); ism1(r) && return PyPtr(); r!=0)
#         if (r=PySetABC_Check(xo); ism1(r) && return PyPtr(); r!=0)
#             gop(xo, yo)
#         else
#             PyNotImplemented_New()
#         end
#     else
#         PyNotImplemented_New()
#     end
# end
# pyjlset_or(xo, yo) = pyjlset_binop_operator(xo, yo, pyjlset_or_generic, pyjlset_or_special)
# pyjlset_xor(xo, yo) = pyjlset_binop_operator(xo, yo, pyjlset_xor_generic, pyjlset_xor_special)
# pyjlset_and(xo, yo) = pyjlset_binop_operator(xo, yo, pyjlset_and_generic, pyjlset_and_special)
# pyjlset_sub(xo, yo) = pyjlset_binop_operator(xo, yo, pyjlset_sub_generic, pyjlset_sub_special)

pyjlset_binop_named(xo, yo, gop, sop) = begin
    t = PyJuliaSetValue_Type()
    isnull(t) && return PyPtr()
    if (r=PyObject_IsInstance(yo, t); ism1(r) && return PyPtr(); r!=0)
        sop(xo, yo)
    elseif (r=PySetABC_Check(yo); ism1(r) && return PyPtr(); r!=0)
        gop(xo, yo)
    else
        PyNotImplemented_New()
    end
end
pyjlset_union(xo, yo) = pyjlset_binop_named(xo, yo, pyjlset_or_generic, pyjlset_or_special)
pyjlset_symmetric_difference(xo, yo) = pyjlset_binop_named(xo, yo, pyjlset_xor_generic, pyjlset_xor_special)
pyjlset_intersection(xo, yo) = pyjlset_binop_named(xo, yo, pyjlset_and_generic, pyjlset_and_special)
pyjlset_difference(xo, yo) = pyjlset_binop_named(xo, yo, pyjlset_sub_generic, pyjlset_sub_special)

pyjlset_isdisjoint(x::PyPtr, y::PyPtr) = _pyjlset_isdisjoint(PyJuliaValue_GetValue(x)::AbstractSet, y)
_pyjlset_isdisjoint(x::AbstractSet, yso::PyPtr) = begin
    isempty(x) && return PyObject_From(true)
    it = PyObject_GetIter(yso)
    isnull(it) && return PyPtr()
    try
        while true
            yo = PyIter_Next(it)
            if !isnull(yo)
                r = PyObject_TryConvert(yo, eltype(x))
                Py_DecRef(yo)
                r == -1 && return PyPtr()
                r ==  0 && continue
                y = takeresult(eltype(x))
                y in x && return PyObject_From(false)
            elseif PyErr_IsSet()
                return PyPtr()
            else
                return PyObject_From(true)
            end
        end
    catch err
        PyErr_SetJuliaError(err)
        return PyPtr()
    finally
        Py_DecRef(it)
    end
end
