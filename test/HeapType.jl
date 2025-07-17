@testitem "heap type from spec" begin
    using PythonCall
    using PythonCall.C

    function _ht_answer(self::C.PyPtr, args::C.PyPtr)
        return C.PyLong_FromLongLong(42)
    end

    meths = C.PyMethodDef[
        C.PyMethodDef(
            name = pointer("answer"),
            meth = @cfunction(_ht_answer, C.PyPtr, (C.PyPtr, C.PyPtr)),
            flags = C.Py_METH_NOARGS,
        ),
        C.PyMethodDef(),
    ]

    slots = C.PyType_Slot[
        C.PyType_Slot(slot = C.Py_tp_methods, pfunc = pointer(meths)),
        C.PyType_Slot(slot = C.Py_tp_new, pfunc = C.POINTERS.PyType_GenericNew),
        C.PyType_Slot(),
    ]

    spec = C.PyType_Spec(
        name = pointer("juliacall.HeapType"),
        basicsize = sizeof(C.PyObject),
        itemsize = 0,
        flags = C.Py_TPFLAGS_DEFAULT | C.Py_TPFLAGS_BASETYPE,
        slots = pointer(slots),
    )

    spec_ref = Ref(spec)
    typ_ptr = GC.@preserve spec slots meths spec_ref begin
        C.PyType_FromSpec(Base.unsafe_convert(Ptr{C.PyType_Spec}, spec_ref))
    end
    @test typ_ptr != C.PyNULL
    typ = PythonCall.pynew(typ_ptr)

    obj = pycall(typ)
    ans = pyconvert(Int, pycall(pygetattr(obj, "answer")))
    @test ans == 42

    PythonCall.pydel!(obj)
    PythonCall.pydel!(typ)
end
