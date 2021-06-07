@enum PyGILState_STATE::Cint PyGILState_LOCKED = 0 PyGILState_UNLOCKED = 1

const Py_single_input = 256
const Py_file_input = 257
const Py_eval_input = 258
const Py_func_type_input = 345

const Py_LT = Cint(0)
const Py_LE = Cint(1)
const Py_EQ = Cint(2)
const Py_NE = Cint(3)
const Py_GT = Cint(4)
const Py_GE = Cint(5)

const Py_METH_VARARGS = 0x0001 # args are a tuple of arguments
const Py_METH_KEYWORDS = 0x0002  # two arguments: the varargs and the kwargs
const Py_METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const Py_METH_O = 0x0008       # single argument (not wrapped in tuple)
const Py_METH_CLASS = 0x0010 # for class methods
const Py_METH_STATIC = 0x0020 # for static methods

const Py_T_SHORT = 0
const Py_T_INT = 1
const Py_T_LONG = 2
const Py_T_FLOAT = 3
const Py_T_DOUBLE = 4
const Py_T_STRING = 5
const Py_T_OBJECT = 6
const Py_T_CHAR = 7
const Py_T_BYTE = 8
const Py_T_UBYTE = 9
const Py_T_USHORT = 10
const Py_T_UINT = 11
const Py_T_ULONG = 12
const Py_T_STRING_INPLACE = 13
const Py_T_BOOL = 14
const Py_T_OBJECT_EX = 16
const Py_T_LONGLONG = 17 # added in Python 2.5
const Py_T_ULONGLONG = 18 # added in Python 2.5
const Py_T_PYSSIZET = 19 # added in Python 2.6
const Py_T_NONE = 20 # added in Python 3.0

const Py_READONLY = 1
const Py_READ_RESTRICTED = 2
const Py_WRITE_RESTRICTED = 4
const Py_RESTRICTED = (Py_READ_RESTRICTED | Py_WRITE_RESTRICTED)

const PyBUF_MAX_NDIM = 64

# Flags for getting buffers
const PyBUF_SIMPLE = 0x0
const PyBUF_WRITABLE = 0x0001
const PyBUF_WRITEABLE = PyBUF_WRITABLE
const PyBUF_FORMAT = 0x0004
const PyBUF_ND = 0x0008
const PyBUF_STRIDES = (0x0010 | PyBUF_ND)
const PyBUF_C_CONTIGUOUS = (0x0020 | PyBUF_STRIDES)
const PyBUF_F_CONTIGUOUS = (0x0040 | PyBUF_STRIDES)
const PyBUF_ANY_CONTIGUOUS = (0x0080 | PyBUF_STRIDES)
const PyBUF_INDIRECT = (0x0100 | PyBUF_STRIDES)

const PyBUF_CONTIG = (PyBUF_ND | PyBUF_WRITABLE)
const PyBUF_CONTIG_RO = (PyBUF_ND)

const PyBUF_STRIDED = (PyBUF_STRIDES | PyBUF_WRITABLE)
const PyBUF_STRIDED_RO = (PyBUF_STRIDES)

const PyBUF_RECORDS = (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_RECORDS_RO = (PyBUF_STRIDES | PyBUF_FORMAT)

const PyBUF_FULL = (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_FULL_RO = (PyBUF_INDIRECT | PyBUF_FORMAT)

const PyBUF_READ = 0x100
const PyBUF_WRITE = 0x200

# Python 2.7
const Py_TPFLAGS_HAVE_GETCHARBUFFER = (0x00000001 << 0)
const Py_TPFLAGS_HAVE_SEQUENCE_IN = (0x00000001 << 1)
const Py_TPFLAGS_GC = 0 # was sometimes (0x00000001<<2) in Python <= 2.1
const Py_TPFLAGS_HAVE_INPLACEOPS = (0x00000001 << 3)
const Py_TPFLAGS_CHECKTYPES = (0x00000001 << 4)
const Py_TPFLAGS_HAVE_RICHCOMPARE = (0x00000001 << 5)
const Py_TPFLAGS_HAVE_WEAKREFS = (0x00000001 << 6)
const Py_TPFLAGS_HAVE_ITER = (0x00000001 << 7)
const Py_TPFLAGS_HAVE_CLASS = (0x00000001 << 8)
const Py_TPFLAGS_HAVE_INDEX = (0x00000001 << 17)
const Py_TPFLAGS_HAVE_NEWBUFFER = (0x00000001 << 21)
const Py_TPFLAGS_STRING_SUBCLASS = (0x00000001 << 27)

# Python 3.0+ has only these:
const Py_TPFLAGS_HEAPTYPE = (0x00000001 << 9)
const Py_TPFLAGS_BASETYPE = (0x00000001 << 10)
const Py_TPFLAGS_READY = (0x00000001 << 12)
const Py_TPFLAGS_READYING = (0x00000001 << 13)
const Py_TPFLAGS_HAVE_GC = (0x00000001 << 14)
const Py_TPFLAGS_HAVE_VERSION_TAG = (0x00000001 << 18)
const Py_TPFLAGS_VALID_VERSION_TAG = (0x00000001 << 19)
const Py_TPFLAGS_IS_ABSTRACT = (0x00000001 << 20)
const Py_TPFLAGS_INT_SUBCLASS = (0x00000001 << 23)
const Py_TPFLAGS_LONG_SUBCLASS = (0x00000001 << 24)
const Py_TPFLAGS_LIST_SUBCLASS = (0x00000001 << 25)
const Py_TPFLAGS_TUPLE_SUBCLASS = (0x00000001 << 26)
const Py_TPFLAGS_BYTES_SUBCLASS = (0x00000001 << 27)
const Py_TPFLAGS_UNICODE_SUBCLASS = (0x00000001 << 28)
const Py_TPFLAGS_DICT_SUBCLASS = (0x00000001 << 29)
const Py_TPFLAGS_BASE_EXC_SUBCLASS = (0x00000001 << 30)
const Py_TPFLAGS_TYPE_SUBCLASS = (0x00000001 << 31)

# only use this if we have the stackless extension
const Py_TPFLAGS_HAVE_STACKLESS_EXTENSION = (0x00000003 << 15)

const Py_hash_t = Cssize_t
const Py_ssize_t = Cssize_t

@kwdef struct Py_complex
    real::Cdouble = 0.0
    imag::Cdouble = 0.0
end

@kwdef struct PyObject
    # assumes _PyObject_HEAD_EXTRA is empty
    refcnt::Py_ssize_t = 0
    type::Ptr{Cvoid} = C_NULL # really is Ptr{PyObject} or Ptr{PyTypeObject} but Julia 1.3 and below get the layout incorrect when circular types are involved
end

const PyPtr = Ptr{PyObject}
const PyNULL = PyPtr(0)

struct PyObjectRef
    ptr::PyPtr
end
ispyreftype(::Type{PyObjectRef}) = true
pyptr(o::PyObjectRef) = o.ptr
Base.unsafe_convert(::Type{PyPtr}, o::PyObjectRef) = o.ptr

@kwdef struct PyVarObject
    ob_base::PyObject = PyObject()
    size::Py_ssize_t = 0
end

@kwdef struct PyMethodDef
    name::Cstring = C_NULL
    meth::Ptr{Cvoid} = C_NULL
    flags::Cint = 0
    doc::Cstring = C_NULL
end

@kwdef struct PyGetSetDef
    name::Cstring = C_NULL
    get::Ptr{Cvoid} = C_NULL
    set::Ptr{Cvoid} = C_NULL
    doc::Cstring = C_NULL
    closure::Ptr{Cvoid} = C_NULL
end

@kwdef struct PyMemberDef
    name::Cstring = C_NULL
    typ::Cint = 0
    offset::Py_ssize_t = 0
    flags::Cint = 0
    doc::Cstring = C_NULL
end

@kwdef struct PyNumberMethods
    add::Ptr{Cvoid} = C_NULL # (o,o)->o
    subtract::Ptr{Cvoid} = C_NULL # (o,o)->o
    multiply::Ptr{Cvoid} = C_NULL # (o,o)->o
    remainder::Ptr{Cvoid} = C_NULL # (o,o)->o
    divmod::Ptr{Cvoid} = C_NULL # (o,o)->o
    power::Ptr{Cvoid} = C_NULL # (o,o,o)->o
    negative::Ptr{Cvoid} = C_NULL # (o)->o
    positive::Ptr{Cvoid} = C_NULL # (o)->o
    absolute::Ptr{Cvoid} = C_NULL # (o)->o
    bool::Ptr{Cvoid} = C_NULL # (o)->Cint
    invert::Ptr{Cvoid} = C_NULL # (o)->o
    lshift::Ptr{Cvoid} = C_NULL # (o,o)->o
    rshift::Ptr{Cvoid} = C_NULL # (o,o)->o
    and::Ptr{Cvoid} = C_NULL # (o,o)->o
    xor::Ptr{Cvoid} = C_NULL # (o,o)->o
    or::Ptr{Cvoid} = C_NULL # (o,o)->o
    int::Ptr{Cvoid} = C_NULL # (o)->o
    _reserved::Ptr{Cvoid} = C_NULL
    float::Ptr{Cvoid} = C_NULL # (o)->o
    inplace_add::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_subtract::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_multiply::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_remainder::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_power::Ptr{Cvoid} = C_NULL # (o,o,o)->o
    inplace_lshift::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_rshift::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_and::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_xor::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_or::Ptr{Cvoid} = C_NULL # (o,o)->o
    floordivide::Ptr{Cvoid} = C_NULL # (o,o)->o
    truedivide::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_floordivide::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_truedivide::Ptr{Cvoid} = C_NULL # (o,o)->o
    index::Ptr{Cvoid} = C_NULL # (o)->o
    matrixmultiply::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_matrixmultiply::Ptr{Cvoid} = C_NULL # (o,o)->o
end

@kwdef struct PySequenceMethods
    length::Ptr{Cvoid} = C_NULL # (o)->Py_ssize_t
    concat::Ptr{Cvoid} = C_NULL # (o,o)->o
    repeat::Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
    item::Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
    _was_item::Ptr{Cvoid} = C_NULL
    ass_item::Ptr{Cvoid} = C_NULL # (o,Py_ssize_t,o)->Cint
    _was_ass_slice::Ptr{Cvoid} = C_NULL
    contains::Ptr{Cvoid} = C_NULL # (o,o)->Cint
    inplace_concat::Ptr{Cvoid} = C_NULL # (o,o)->o
    inplace_repeat::Ptr{Cvoid} = C_NULL # (o,Py_ssize_t)->o
end

@kwdef struct PyMappingMethods
    length::Ptr{Cvoid} = C_NULL # (o)->Py_ssize_t
    subscript::Ptr{Cvoid} = C_NULL # (o,o)->o
    ass_subscript::Ptr{Cvoid} = C_NULL # (o,o,o)->Cint
end

@kwdef struct PyBufferProcs
    get::Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}, Cint) -> Cint
    release::Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}) -> Cvoid
end

@kwdef struct Py_buffer
    buf::Ptr{Cvoid} = C_NULL
    obj::Ptr{Cvoid} = C_NULL
    len::Py_ssize_t = 0
    itemsize::Py_ssize_t = 0
    readonly::Cint = 0
    ndim::Cint = 0
    format::Cstring = C_NULL
    shape::Ptr{Py_ssize_t} = C_NULL
    strides::Ptr{Py_ssize_t} = C_NULL
    suboffsets::Ptr{Py_ssize_t} = C_NULL
    internal::Ptr{Cvoid} = C_NULL
end

@kwdef struct PyTypeObject
    ob_base::PyVarObject = PyVarObject()
    name::Cstring = C_NULL

    basicsize::Py_ssize_t = 0
    itemsize::Py_ssize_t = 0

    dealloc::Ptr{Cvoid} = C_NULL
    vectorcall_offset::Py_ssize_t = 0
    getattr::Ptr{Cvoid} = C_NULL
    setattr::Ptr{Cvoid} = C_NULL
    as_async::Ptr{Cvoid} = C_NULL
    repr::Ptr{Cvoid} = C_NULL

    as_number::Ptr{PyNumberMethods} = C_NULL
    as_sequence::Ptr{PySequenceMethods} = C_NULL
    as_mapping::Ptr{PyMappingMethods} = C_NULL

    hash::Ptr{Cvoid} = C_NULL
    call::Ptr{Cvoid} = C_NULL
    str::Ptr{Cvoid} = C_NULL
    getattro::Ptr{Cvoid} = C_NULL
    setattro::Ptr{Cvoid} = C_NULL

    as_buffer::Ptr{PyBufferProcs} = C_NULL

    flags::Culong = 0

    doc::Cstring = C_NULL

    traverse::Ptr{Cvoid} = C_NULL

    clear::Ptr{Cvoid} = C_NULL

    richcompare::Ptr{Cvoid} = C_NULL

    weaklistoffset::Py_ssize_t = 0

    iter::Ptr{Cvoid} = C_NULL
    iternext::Ptr{Cvoid} = C_NULL

    methods::Ptr{PyMethodDef} = C_NULL
    members::Ptr{PyMemberDef} = C_NULL
    getset::Ptr{PyGetSetDef} = C_NULL
    base::PyPtr = C_NULL
    dict::PyPtr = C_NULL
    descr_get::Ptr{Cvoid} = C_NULL
    descr_set::Ptr{Cvoid} = C_NULL
    dictoffset::Py_ssize_t = 0
    init::Ptr{Cvoid} = C_NULL
    alloc::Ptr{Cvoid} = C_NULL
    new::Ptr{Cvoid} = C_NULL
    free::Ptr{Cvoid} = C_NULL
    is_gc::Ptr{Cvoid} = C_NULL
    bases::PyPtr = C_NULL
    mro::PyPtr = C_NULL
    cache::PyPtr = C_NULL
    subclasses::PyPtr = C_NULL
    weaklist::PyPtr = C_NULL
    del::Ptr{Cvoid} = C_NULL

    version_tag::Cuint = 0

    finalize::Ptr{Cvoid} = C_NULL
    vectorcall::Ptr{Cvoid} = C_NULL

end

const PyTypePtr = Ptr{PyTypeObject}

struct PySimpleObject{T}
    ob_base::PyObject
    value::T
end

PySimpleObject_GetValue(__o, ::Type{T}) where {T} = begin
    _o = Base.cconvert(PyPtr, __o)
    GC.@preserve _o begin
        o = Base.unsafe_convert(PyPtr, _o)
        UnsafePtr{PySimpleObject{T}}(o).value[!]::T
    end
end
