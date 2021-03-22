pyjlset_add(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    ism1(PyObject_Convert(vo, eltype(x))) && return PyNULL
    v = takeresult(eltype(x))
    @pyjltry begin
        push!(x, v)
        PyNone_New()
    end PyNULL
end

pyjlset_discard(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    r = PyObject_TryConvert(vo, eltype(x))
    r == -1 && return PyNULL
    r == 0 && return PyNone_New()
    v = takeresult(eltype(x))
    @pyjltry begin
        delete!(x, v)
        PyNone_New()
    end PyNULL
end

pyjlset_clear(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    @pyjltry begin
        empty!(x)
        PyNone_New()
    end PyNULL
end

pyjlset_copy(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    @pyjltry PyObject_From(copy(x)) PyNULL
end

pyjlset_pop(xo::PyPtr, _::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    @pyjltry begin
        if isempty(x)
            PyErr_SetString(PyExc_KeyError(), "pop from an empty set")
            PyNULL
        else
            PyObject_From(pop!(x))
        end
    end PyNULL
end

pyjlset_remove(xo::PyPtr, vo::PyPtr) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    r = PyObject_TryConvert(vo, eltype(x))
    if r == -1
        return PyNULL
    elseif r == 0
        PyErr_SetObject(PyExc_KeyError(), vo)
        return PyNULL
    end
    v = takeresult(eltype(x))
    @pyjltry begin
        if v in x
            delete!(x, v)
            PyNone_New()
        else
            PyErr_SetObject(PyExc_KeyError(), vo)
            PyNULL
        end
    end PyNULL
end

pyjlset_ibinop_named(xo, yo, op, skip) = begin
    x = PyJuliaValue_GetValue(xo)::AbstractSet
    y = PyIterable_Collect(yo, eltype(x), skip)
    isempty(y) && PyErr_IsSet() && return PyNULL
    @pyjltry begin
        op(x, y)
        PyNone_New()
    end PyNULL
end
pyjlset_update(xo, yo) = pyjlset_ibinop_named(xo, yo, union!, false)
pyjlset_difference_update(xo, yo) = pyjlset_ibinop_named(xo, yo, setdiff!, true)
pyjlset_symmetric_difference_update(xo, yo) = pyjlset_ibinop_named(xo, yo, symdiff!, false)
pyjlset_intersection_update(xo, yo) = pyjlset_ibinop_named(xo, yo, intersect!, true)

# pyjlset_ibinop_operator(xo, yo, op, skip) = begin
#     r = PySetABC_Check(yo)
#     r == -1 && return PyNULL
#     r ==  0 && return PyNotImplemented_New()
#     x = PyJuliaValue_GetValue(xo)::AbstractSet
#     y = PyIterable_Collect(yo, eltype(x), skip)
#     isempty(y) && PyErr_IsSet() && return PyNULL
#     @pyjltry PyObject_From(op(x, y)) PyNULL
# end
# pyjlset_ior(xo, yo) = pyjlset_ibinop_operator(xo, yo, union!, false)
# pyjlset_isub(xo, yo) = pyjlset_ibinop_operator(xo, yo, setdiff!, true)
# pyjlset_ixor(xo, yo) = pyjlset_ibinop_operator(xo, yo, symdiff!, false)
# pyjlset_iand(xo, yo) = pyjlset_ibinop_operator(xo, yo, intersect!, true)

pyjlset_binop_generic(xo::PyPtr, yo::PyPtr, op) = begin
    xo2 = PySet_New(xo)
    isnull(xo2) && return PyNULL
    yo2 = PySet_New(yo)
    isnull(yo2) && (Py_DecRef(xo2); return PyNULL)
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
    @pyjltry PyObject_From(op(x, y)) PyNULL
end
pyjlset_or_special(xo, yo) = pyjlset_binop_special(xo, yo, union)
pyjlset_xor_special(xo, yo) = pyjlset_binop_special(xo, yo, symdiff)
pyjlset_and_special(xo, yo) = pyjlset_binop_special(xo, yo, intersect)
pyjlset_sub_special(xo, yo) = pyjlset_binop_special(xo, yo, setdiff)

# pyjlset_binop_operator(xo::PyPtr, yo::PyPtr, gop, sop) = begin
#     t = PyJuliaSetValue_Type()
#     isnull(t) && return PyNULL
#     if (r=PyObject_IsInstance(xo, t); ism1(r) && return PyNULL; r!=0)
#         if (r=PyObject_IsInstance(yo, t); ism1(r) && return PyNULL; r!=0)
#             sop(xo, yo)
#         elseif (r=PySetABC_Check(yo); ism1(r) && return PyNULL; r!=0)
#             gop(xo, yo)
#         else
#             PyNotImplemented_New()
#         end
#     elseif (r=PyObject_IsInstance(yo, t); ism1(r) && return PyNULL; r!=0)
#         if (r=PySetABC_Check(xo); ism1(r) && return PyNULL; r!=0)
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
    isnull(t) && return PyNULL
    if (r = PyObject_IsInstance(yo, t); ism1(r) && return PyNULL; r != 0)
        sop(xo, yo)
    elseif (r = PySetABC_Check(yo); ism1(r) && return PyNULL; r != 0)
        gop(xo, yo)
    else
        PyNotImplemented_New()
    end
end
pyjlset_union(xo, yo) = pyjlset_binop_named(xo, yo, pyjlset_or_generic, pyjlset_or_special)
pyjlset_symmetric_difference(xo, yo) =
    pyjlset_binop_named(xo, yo, pyjlset_xor_generic, pyjlset_xor_special)
pyjlset_intersection(xo, yo) =
    pyjlset_binop_named(xo, yo, pyjlset_and_generic, pyjlset_and_special)
pyjlset_difference(xo, yo) =
    pyjlset_binop_named(xo, yo, pyjlset_sub_generic, pyjlset_sub_special)

pyjlset_isdisjoint(x::PyPtr, y::PyPtr) =
    _pyjlset_isdisjoint(PyJuliaValue_GetValue(x)::AbstractSet, y)
_pyjlset_isdisjoint(x::AbstractSet, yso::PyPtr) = begin
    isempty(x) && return PyObject_From(true)
    r = PyIterable_Map(yso) do yo
        r = PyObject_TryConvert(yo, eltype(x))
        r == -1 && return -1
        r == 0 && return 1
        y == takeresult(eltype(x))
        y in x ? 0 : 1
    end
    r == -1 ? PyNULL : r == 0 ? PyObject_From(false) : PyObject_From(true)
end

const PyJuliaSetValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.SetValue"),
        base = base,
        methods = cacheptr!(c, [
            PyMethodDef(
                name = cacheptr!(c, "add"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_add),
            ),
            PyMethodDef(
                name = cacheptr!(c, "clear"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjlset_clear),
            ),
            PyMethodDef(
                name = cacheptr!(c, "copy"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjlset_copy),
            ),
            PyMethodDef(
                name = cacheptr!(c, "difference"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjlset_difference),
            ),
            PyMethodDef(
                name = cacheptr!(c, "difference_update"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_difference_update),
            ),
            PyMethodDef(
                name = cacheptr!(c, "discard"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_discard),
            ),
            PyMethodDef(
                name = cacheptr!(c, "intersection"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_intersection),
            ),
            PyMethodDef(
                name = cacheptr!(c, "intersection_update"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_intersection_update),
            ),
            PyMethodDef(
                name = cacheptr!(c, "isdisjoint"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_isdisjoint),
            ),
            # PyMethodDef(
            #     name = cacheptr!(c, "issubset"),
            #     flags = Py_METH_O,
            #     meth = @cfunctionOO(pyjlset_issubset),
            # ),
            # PyMethodDef(
            #     name = cacheptr!(c, "issuperset"),
            #     flags = Py_METH_O,
            #     meth = @cfunctionOO(pyjlset_issuperset),
            # ),
            PyMethodDef(
                name = cacheptr!(c, "pop"),
                flags = Py_METH_NOARGS,
                meth = @cfunctionOO(pyjlset_pop),
            ),
            PyMethodDef(
                name = cacheptr!(c, "remove"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_remove),
            ),
            PyMethodDef(
                name = cacheptr!(c, "symmetric_difference"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_symmetric_difference),
            ),
            PyMethodDef(
                name = cacheptr!(c, "symmetric_difference_update"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_symmetric_difference_update),
            ),
            PyMethodDef(
                name = cacheptr!(c, "union"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_union),
            ),
            PyMethodDef(
                name = cacheptr!(c, "update"),
                flags = Py_METH_O,
                meth = @cfunctionOO(pyjlset_update),
            ),
            PyMethodDef(),
        ]),
        # as_number = cacheptr!(c, fill(PyNumberMethods(
        #     or = @cfunctionOO(pyjlset_or),
        #     and = @cfunctionOO(pyjlset_and),
        #     xor = @cfunctionOO(pyjlset_xor),
        #     subtract = @cfunctionOO(pyjlset_subtract),
        #     inplace_or = @cfunctionOO(pyjlset_ior),
        #     inplace_and = @cfunctionOO(pyjlset_iand),
        #     inplace_xor = @cfunctionOO(pyjlset_ixor),
        #     inplace_subtract = @cfunctionOO(pyjlset_isubtract),
        # ))),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    abc = PyMutableSetABC_Type()
    isnull(abc) && return PyNULL
    ism1(PyABC_Register(ptr, abc)) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaSetValue_New(x::AbstractSet) = PyJuliaValue_New(PyJuliaSetValue_Type(), x)
PyJuliaValue_From(x::AbstractSet) = PyJuliaSetValue_New(x)
