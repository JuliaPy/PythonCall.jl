@kwdef struct CPyNumberMethodsStruct
    add :: Ptr{Cvoid} = C_NULL # (o,o)->o
    subtract :: Ptr{Cvoid} = C_NULL # (o,o)->o
    multiply :: Ptr{Cvoid} = C_NULL # (o,o)->o
    remainder :: Ptr{Cvoid} = C_NULL # (o,o)->o
    divmod :: Ptr{Cvoid} = C_NULL # (o,o)->o
    power :: Ptr{Cvoid} = C_NULL # (o,o,o)->o
    negative :: Ptr{Cvoid} = C_NULL # (o)->o
    positive :: Ptr{Cvoid} = C_NULL # (o)->o
    absolute :: Ptr{Cvoid} = C_NULL # (o)->o
    bool :: Ptr{Cvoid} = C_NULL # (o)->Cint
    invert :: Ptr{Cvoid} = C_NULL # (o)->o
    lshift :: Ptr{Cvoid} = C_NULL # (o,o)->o
    rshift :: Ptr{Cvoid} = C_NULL # (o,o)->o
    and :: Ptr{Cvoid} = C_NULL # (o,o)->o
    xor :: Ptr{Cvoid} = C_NULL # (o,o)->o
    or :: Ptr{Cvoid} = C_NULL # (o,o)->o
    int :: Ptr{Cvoid} = C_NULL # (o)->o
    _reserved :: Ptr{Cvoid} = C_NULL
    float :: Ptr{Cvoid} = C_NULL # (o)->o
    inplace_add :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_subtract :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_multiply :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_remainder :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_power :: Ptr{Cvoid} = C_NULL # (o,o,o)->o
    inplace_lshift :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_rshift :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_and :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_xor :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_or :: Ptr{Cvoid} = C_NULL # (o,o)->o
    floordivide :: Ptr{Cvoid} = C_NULL # (o,o)->o
    truedivide :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_floordivide :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_truedivide :: Ptr{Cvoid} = C_NULL # (o,o)->o
    index :: Ptr{Cvoid} = C_NULL # (o)->o
    matrixmultiply :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_matrixmultiply :: Ptr{Cvoid} = C_NULL # (o,o)->o
end

@kwdef struct CPySequenceMethodsStruct
    length :: Ptr{Cvoid} = C_NULL # (o)->Py_ssize_t
    concat :: Ptr{Cvoid} = C_NULL # (o,o)->o
    repeat :: Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
    item :: Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
    _was_item :: Ptr{Cvoid} = C_NULL
    ass_item :: Ptr{Cvoid} = C_NULL # (o,Py_ssize_t,o)->Cint
    _was_ass_slice :: Ptr{Cvoid} = C_NULL
    contains :: Ptr{Cvoid} = C_NULL # (o,o)->Cint
    inplace_concat :: Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_repeat :: Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
end

@kwdef struct CPyMappingMethodsStruct
    length :: Ptr{Cvoid} = C_NULL # (o)->Py_ssize_t
    subscript :: Ptr{Cvoid} = C_NULL # (o,o)->o
    ass_subscript :: Ptr{Cvoid} = C_NULL # (o,o,o)->Cint
end
