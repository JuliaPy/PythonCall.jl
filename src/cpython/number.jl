PyNumber_Add(x, y) = ccall(POINTERS.PyNumber_Add, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Subtract(x, y) = ccall(POINTERS.PyNumber_Subtract, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Multiply(x, y) = ccall(POINTERS.PyNumber_Multiply, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_MatrixMultiply(x, y) = ccall(POINTERS.PyNumber_MatrixMultiply, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_FloorDivide(x, y) = ccall(POINTERS.PyNumber_FloorDivide, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_TrueDivide(x, y) = ccall(POINTERS.PyNumber_TrueDivide, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Remainder(x, y) = ccall(POINTERS.PyNumber_Remainder, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Divmod(x, y) = ccall(POINTERS.PyNumber_Divmod, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Power(x, y, z) = ccall(POINTERS.PyNumber_Power, PyPtr, (PyPtr, PyPtr, PyPtr), x, y, z)
PyNumber_Negative(x) = ccall(POINTERS.PyNumber_Negative, PyPtr, (PyPtr,), x)
PyNumber_Positive(x) = ccall(POINTERS.PyNumber_Positive, PyPtr, (PyPtr,), x)
PyNumber_Absolute(x) = ccall(POINTERS.PyNumber_Absolute, PyPtr, (PyPtr,), x)
PyNumber_Invert(x) = ccall(POINTERS.PyNumber_Invert, PyPtr, (PyPtr,), x)
PyNumber_Lshift(x, y) = ccall(POINTERS.PyNumber_Lshift, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Rshift(x, y) = ccall(POINTERS.PyNumber_Rshift, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_And(x, y) = ccall(POINTERS.PyNumber_And, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Xor(x, y) = ccall(POINTERS.PyNumber_Xor, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Or(x, y) = ccall(POINTERS.PyNumber_Or, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceAdd(x, y) = ccall(POINTERS.PyNumber_InPlaceAdd, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceSubtract(x, y) = ccall(POINTERS.PyNumber_InPlaceSubtract, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceMultiply(x, y) = ccall(POINTERS.PyNumber_InPlaceMultiply, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceMatrixMultiply(x, y) = ccall(POINTERS.PyNumber_InPlaceMatrixMultiply, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceFloorDivide(x, y) = ccall(POINTERS.PyNumber_InPlaceFloorDivide, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceTrueDivide(x, y) = ccall(POINTERS.PyNumber_InPlaceTrueDivide, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceRemainder(x, y) = ccall(POINTERS.PyNumber_InPlaceRemainder, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlacePower(x, y, z) = ccall(POINTERS.PyNumber_InPlacePower, PyPtr, (PyPtr, PyPtr, PyPtr), x, y, z)
PyNumber_InPlaceLshift(x, y) = ccall(POINTERS.PyNumber_InPlaceLshift, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceRshift(x, y) = ccall(POINTERS.PyNumber_InPlaceRshift, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceAnd(x, y) = ccall(POINTERS.PyNumber_InPlaceAnd, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceXor(x, y) = ccall(POINTERS.PyNumber_InPlaceXor, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_InPlaceOr(x, y) = ccall(POINTERS.PyNumber_InPlaceOr, PyPtr, (PyPtr, PyPtr), x, y)
PyNumber_Long(x) = ccall(POINTERS.PyNumber_Long, PyPtr, (PyPtr,), x)
PyNumber_Float(x) = ccall(POINTERS.PyNumber_Float, PyPtr, (PyPtr,), x)
PyNumber_Index(x) = ccall(POINTERS.PyNumber_Index, PyPtr, (PyPtr,), x)

for n in [:Number, :Complex, :Real, :Rational, :Integral]
    p = Symbol(:Py, n, :ABC)
    t = Symbol(p, :_Type)
    tr = Symbol(p, :__ref)
    c = Symbol(p, :_Check)
    @eval const $tr = Ref(PyNULL)
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
