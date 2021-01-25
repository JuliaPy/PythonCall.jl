@cdef :PyNumber_Add PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Subtract PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Multiply PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_MatrixMultiply PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_FloorDivide PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_TrueDivide PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Remainder PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_DivMod PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Power PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :PyNumber_Negative PyPtr (PyPtr,)
@cdef :PyNumber_Positive PyPtr (PyPtr,)
@cdef :PyNumber_Absolute PyPtr (PyPtr,)
@cdef :PyNumber_Invert PyPtr (PyPtr,)
@cdef :PyNumber_Lshift PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Rshift PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_And PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Xor PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Or PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceAdd PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceSubtract PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceMultiply PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceMatrixMultiply PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceFloorDivide PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceTrueDivide PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceRemainder PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlacePower PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :PyNumber_InPlaceLshift PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceRshift PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceAnd PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceXor PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_InPlaceOr PyPtr (PyPtr, PyPtr)
@cdef :PyNumber_Long PyPtr (PyPtr,)
@cdef :PyNumber_Float PyPtr (PyPtr,)
@cdef :PyNumber_Index PyPtr (PyPtr,)

for n in [:Number, :Complex, :Real, :Rational, :Integral]
    p = Symbol(:Py, n, :ABC)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyPtr())
    @eval $t(doimport::Bool = true) = begin
        ptr = $tr[]
        isnull(ptr) || return ptr
        a = doimport ? PyImport_ImportModule("numbers") : PyImport_GetModule("numbers")
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
