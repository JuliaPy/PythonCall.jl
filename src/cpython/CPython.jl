module CPython

using Libdl
using ..Python: CONFIG, isnull, ism1, PYERR, NOTIMPLEMENTED, _typeintersect
using Base: @kwdef
using UnsafePointers: UnsafePtr

tryconvert(::Type{T}, x::PYERR) where {T} = PYERR()
tryconvert(::Type{T}, x::NOTIMPLEMENTED) where {T} = NOTIMPLEMENTED()
tryconvert(::Type{T}, x::T) where {T} = x
tryconvert(::Type{T}, x) where {T} =
    try
        convert(T, x)
    catch
        NOTIMPLEMENTED()
    end

@enum PyGILState_STATE::Cint PyGILState_LOCKED=0 PyGILState_UNLOCKED=1

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

const Py_T_SHORT        =0
const Py_T_INT          =1
const Py_T_LONG         =2
const Py_T_FLOAT        =3
const Py_T_DOUBLE       =4
const Py_T_STRING       =5
const Py_T_OBJECT       =6
const Py_T_CHAR         =7
const Py_T_BYTE         =8
const Py_T_UBYTE        =9
const Py_T_USHORT       =10
const Py_T_UINT         =11
const Py_T_ULONG        =12
const Py_T_STRING_INPLACE       =13
const Py_T_BOOL         =14
const Py_T_OBJECT_EX    =16
const Py_T_LONGLONG     =17 # added in Python 2.5
const Py_T_ULONGLONG    =18 # added in Python 2.5
const Py_T_PYSSIZET     =19 # added in Python 2.6
const Py_T_NONE         =20 # added in Python 3.0

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
const Py_TPFLAGS_HAVE_GETCHARBUFFER  = (0x00000001<<0)
const Py_TPFLAGS_HAVE_SEQUENCE_IN = (0x00000001<<1)
const Py_TPFLAGS_GC = 0 # was sometimes (0x00000001<<2) in Python <= 2.1
const Py_TPFLAGS_HAVE_INPLACEOPS = (0x00000001<<3)
const Py_TPFLAGS_CHECKTYPES = (0x00000001<<4)
const Py_TPFLAGS_HAVE_RICHCOMPARE = (0x00000001<<5)
const Py_TPFLAGS_HAVE_WEAKREFS = (0x00000001<<6)
const Py_TPFLAGS_HAVE_ITER = (0x00000001<<7)
const Py_TPFLAGS_HAVE_CLASS = (0x00000001<<8)
const Py_TPFLAGS_HAVE_INDEX = (0x00000001<<17)
const Py_TPFLAGS_HAVE_NEWBUFFER = (0x00000001<<21)
const Py_TPFLAGS_STRING_SUBCLASS       = (0x00000001<<27)

# Python 3.0+ has only these:
const Py_TPFLAGS_HEAPTYPE = (0x00000001<<9)
const Py_TPFLAGS_BASETYPE = (0x00000001<<10)
const Py_TPFLAGS_READY = (0x00000001<<12)
const Py_TPFLAGS_READYING = (0x00000001<<13)
const Py_TPFLAGS_HAVE_GC = (0x00000001<<14)
const Py_TPFLAGS_HAVE_VERSION_TAG   = (0x00000001<<18)
const Py_TPFLAGS_VALID_VERSION_TAG  = (0x00000001<<19)
const Py_TPFLAGS_IS_ABSTRACT = (0x00000001<<20)
const Py_TPFLAGS_INT_SUBCLASS         = (0x00000001<<23)
const Py_TPFLAGS_LONG_SUBCLASS        = (0x00000001<<24)
const Py_TPFLAGS_LIST_SUBCLASS        = (0x00000001<<25)
const Py_TPFLAGS_TUPLE_SUBCLASS       = (0x00000001<<26)
const Py_TPFLAGS_BYTES_SUBCLASS       = (0x00000001<<27)
const Py_TPFLAGS_UNICODE_SUBCLASS     = (0x00000001<<28)
const Py_TPFLAGS_DICT_SUBCLASS        = (0x00000001<<29)
const Py_TPFLAGS_BASE_EXC_SUBCLASS    = (0x00000001<<30)
const Py_TPFLAGS_TYPE_SUBCLASS        = (0x00000001<<31)

# only use this if we have the stackless extension
const Py_TPFLAGS_HAVE_STACKLESS_EXTENSION = (0x00000003<<15)

const Py_hash_t = Cssize_t
const Py_ssize_t = Cssize_t

@kwdef struct Py_complex
    real :: Cdouble = 0.0
    imag :: Cdouble = 0.0
end

@kwdef struct PyObject
    # assumes _PyObject_HEAD_EXTRA is empty
    refcnt :: Py_ssize_t = 0
    type :: Ptr{PyObject} = C_NULL
end

const PyPtr = Ptr{PyObject}

struct PyObjectRef
    ptr :: PyPtr
end
Base.unsafe_convert(::Type{PyPtr}, o::PyObjectRef) = o.ptr

@kwdef struct PyVarObject
    ob_base :: PyObject = PyObject()
    size :: Py_ssize_t = 0
end

@kwdef struct PyMethodDef
    name :: Cstring = C_NULL
    meth :: Ptr{Cvoid} = C_NULL
    flags :: Cint = 0
    doc :: Cstring = C_NULL
end

@kwdef struct PyGetSetDef
    name :: Cstring = C_NULL
    get :: Ptr{Cvoid} = C_NULL
    set :: Ptr{Cvoid} = C_NULL
    doc :: Cstring = C_NULL
    closure :: Ptr{Cvoid} = C_NULL
end

@kwdef struct PyMemberDef
    name :: Cstring = C_NULL
    typ :: Cint = 0
    offset :: Py_ssize_t = 0
    flags :: Cint = 0
    doc :: Cstring = C_NULL
end

@kwdef struct PyNumberMethods
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

@kwdef struct PySequenceMethods
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

@kwdef struct PyMappingMethods
    length :: Ptr{Cvoid} = C_NULL # (o)->Py_ssize_t
    subscript :: Ptr{Cvoid} = C_NULL # (o,o)->o
    ass_subscript :: Ptr{Cvoid} = C_NULL # (o,o,o)->Cint
end

@kwdef struct PyBufferProcs
    get :: Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}, Cint) -> Cint
    release :: Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}) -> Cvoid
end

@kwdef struct Py_buffer
    buf :: Ptr{Cvoid} = C_NULL
    obj :: Ptr{Cvoid} = C_NULL
    len :: Py_ssize_t = 0
    itemsize :: Py_ssize_t = 0
    readonly :: Cint = 0
    ndim :: Cint = 0
    format :: Cstring = C_NULL
    shape :: Ptr{Py_ssize_t} = C_NULL
    strides :: Ptr{Py_ssize_t} = C_NULL
    suboffsets :: Ptr{Py_ssize_t} = C_NULL
    internal :: Ptr{Cvoid} = C_NULL
end

@kwdef struct PyTypeObject
    ob_base :: PyVarObject = PyVarObject()
    name :: Cstring = C_NULL

    basicsize :: Py_ssize_t = 0
    itemsize :: Py_ssize_t = 0

    dealloc :: Ptr{Cvoid} = C_NULL
    vectorcall_offset :: Py_ssize_t = 0
    getattr :: Ptr{Cvoid} = C_NULL
    setattr :: Ptr{Cvoid} = C_NULL
    as_async :: Ptr{Cvoid} = C_NULL
    repr :: Ptr{Cvoid} = C_NULL

    as_number :: Ptr{PyNumberMethods} = C_NULL
    as_sequence :: Ptr{PySequenceMethods} = C_NULL
    as_mapping :: Ptr{PyMappingMethods} = C_NULL

    hash :: Ptr{Cvoid} = C_NULL
    call :: Ptr{Cvoid} = C_NULL
    str :: Ptr{Cvoid} = C_NULL
    getattro :: Ptr{Cvoid} = C_NULL
    setattro :: Ptr{Cvoid} = C_NULL

    as_buffer :: Ptr{PyBufferProcs} = C_NULL

    flags :: Culong = 0

    doc :: Cstring = C_NULL

    traverse :: Ptr{Cvoid} = C_NULL

    clear :: Ptr{Cvoid} = C_NULL

    richcompare :: Ptr{Cvoid} = C_NULL

    weaklistoffset :: Py_ssize_t = 0

    iter :: Ptr{Cvoid} = C_NULL
    iternext :: Ptr{Cvoid} = C_NULL

    methods :: Ptr{PyMethodDef} = C_NULL
    members :: Ptr{PyMemberDef} = C_NULL
    getset :: Ptr{PyGetSetDef} = C_NULL
    base :: PyPtr = C_NULL
    dict :: PyPtr = C_NULL
    descr_get :: Ptr{Cvoid} = C_NULL
    descr_set :: Ptr{Cvoid} = C_NULL
    dictoffset :: Py_ssize_t = 0
    init :: Ptr{Cvoid} = C_NULL
    alloc :: Ptr{Cvoid} = C_NULL
    new :: Ptr{Cvoid} = C_NULL
    free :: Ptr{Cvoid} = C_NULL
    is_gc :: Ptr{Cvoid} = C_NULL
    bases :: PyPtr = C_NULL
    mro :: PyPtr = C_NULL
    cache :: PyPtr = C_NULL
    subclasses :: PyPtr = C_NULL
    weaklist :: PyPtr = C_NULL
    del :: Ptr{Cvoid} = C_NULL

    version_tag :: Cuint = 0

    finalize :: Ptr{Cvoid} = C_NULL
    vectorcall :: Ptr{Cvoid} = C_NULL

end

const PyTypePtr = Ptr{PyTypeObject}

pyglobal(name) = dlsym(CONFIG.libptr, name)
pyglobal(r::Ref{Ptr{Cvoid}}, name) = (p=r[]; if isnull(p); p=r[]=pyglobal(name); end; p)

pyloadglobal(r::Ref{Ptr{T}}, name) where {T} = (p=r[]; if isnull(p); p=r[]=unsafe_load(Ptr{Ptr{T}}(pyglobal(name))) end; p)

macro cdef(name, rettype, argtypes)
    name isa QuoteNode && name.value isa Symbol || error("name must be a symbol, got $name")
    jname = esc(name.value)
    refname = esc(Symbol(name.value, :__ref))
    name = esc(name)
    rettype = esc(rettype)
    argtypes isa Expr && argtypes.head==:tuple || error("argtypes must be a tuple, got $argtypes")
    nargs = length(argtypes.args)
    argtypes = esc(argtypes)
    args = [gensym() for i in 1:nargs]
    quote
        const $refname = Ref(C_NULL)
        $jname($(args...)) = ccall(pyglobal($refname, $name), $rettype, $argtypes, $(args...))
    end
end

### INITIALIZE

@cdef :Py_Initialize Cvoid ()
@cdef :Py_InitializeEx Cvoid (Cint,)
@cdef :Py_Finalize Cvoid ()
@cdef :Py_FinalizeEx Cint ()
@cdef :Py_AtExit Cint (Ptr{Cvoid},)
@cdef :Py_IsInitialized Cint ()

@cdef :Py_SetPythonHome Cvoid (Cwstring,)
@cdef :Py_SetProgramName Cvoid (Cwstring,)
@cdef :Py_GetVersion Cstring ()

### REFCOUNT

@cdef :Py_IncRef Cvoid (PyPtr,)
@cdef :Py_DecRef Cvoid (PyPtr,)
Py_RefCnt(o) = GC.@preserve o UnsafePtr(Base.unsafe_convert(PyPtr, o)).refcnt[]

Py_DecRef(f::Function, o::Ptr, dflt=PYERR()) =
    isnull(o) ? dflt : (r=f(o); Py_DecRef(o); r)

Py_Is(o1, o2) = Base.unsafe_convert(PyPtr, o1) == Base.unsafe_convert(PyPtr, o2)

### EVAL

@cdef :PyEval_EvalCode PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :Py_CompileString PyPtr (Cstring, Cstring, Cint)
@cdef :PyEval_GetBuiltins PyPtr ()

### GIL & THREADS

@cdef :PyEval_SaveThread Ptr{Cvoid} ()
@cdef :PyEval_RestoreThread Cvoid (Ptr{Cvoid},)
@cdef :PyGILState_Ensure PyGILState_STATE ()
@cdef :PyGILState_Release Cvoid (PyGILState_STATE,)

### IMPORT

@cdef :PyImport_ImportModule PyPtr (Cstring,)
@cdef :PyImport_Import PyPtr (PyPtr,)

### ERRORS

@cdef :PyErr_Occurred PyPtr ()
@cdef :PyErr_GivenExceptionMatches Cint (PyPtr, PyPtr)
@cdef :PyErr_Clear Cvoid ()
@cdef :PyErr_SetNone Cvoid (PyPtr,)
@cdef :PyErr_SetString Cvoid (PyPtr, Cstring)
@cdef :PyErr_SetObject Cvoid (PyPtr, PyPtr)
@cdef :PyErr_Fetch Cvoid (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr})
@cdef :PyErr_NormalizeException Cvoid (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr})
@cdef :PyErr_Restore Cvoid (PyPtr, PyPtr, PyPtr)

PyErr_IsSet() = !isnull(PyErr_Occurred())

PyErr_IsSet(t) = (o=PyErr_Occurred(); !isnull(o) && PyErr_GivenExceptionMatches(o,t)!=0)

function PyErr_FetchTuple(normalize::Bool=false)
    t = Ref{PyPtr}()
    v = Ref{PyPtr}()
    b = Ref{PyPtr}()
    PyErr_Fetch(t, v, b)
    normalize && PyErr_NormalizeException(t, v, b)
    (t[], v[], b[])
end

### EXCEPTIONS

for x in [:BaseException, :Exception, :StopIteration, :GeneratorExit, :ArithmeticError,
    :LookupError, :AssertionError, :AttributeError, :BufferError, :EOFError,
    :FloatingPointError, :OSError, :ImportError, :IndexError, :KeyError, :KeyboardInterrupt,
    :MemoryError, :NameError, :OverflowError, :RuntimeError, :RecursionError,
    :NotImplementedError, :SyntaxError, :IndentationError, :TabError, :ReferenceError,
    :SystemError, :SystemExit, :TypeError, :UnboundLocalError, :UnicodeError,
    :UnicodeEncodeError, :UnicodeDecodeError, :UnicodeTranslateError, :ValueError,
    :ZeroDivisionError, :BlockingIOError, :BrokenPipeError, :ChildProcessError,
    :ConnectionError, :ConnectionAbortedError, :ConnectionRefusedError, :FileExistsError,
    :FileNotFoundError, :InterruptedError, :IsADirectoryError, :NotADirectoryError,
    :PermissionError, :ProcessLookupError, :TimeoutError, :EnvironmentError, :IOError,
    :WindowsError, :Warning, :UserWarning, :DeprecationWarning, :PendingDeprecationWarning,
    :SyntaxWarning, :RuntimeWarning, :FutureWarning, :ImportWarning, :UnicodeWarning,
    :BytesWarning, :ResourceWarning]
    f = Symbol(:PyExc_, x)
    r = Symbol(f, :__ref)
    @eval const $r = Ref(PyPtr())
    @eval $f() = pyloadglobal($r, $(QuoteNode(f)))
end

### NONE

const Py_None__ref = Ref(C_NULL)
Py_None() = PyPtr(pyglobal(Py_None__ref, :_Py_NoneStruct))

PyNone_Check(o) = Py_Is(o, Py_None())

PyNone_New() = (o=Py_None(); Py_IncRef(o); o)

PyNone_As(o, ::Type{T}) where {T} =
    if Nothing <: T
        nothing
    elseif Missing <: T
        missing
    else
        NOTIMPLEMENTED()
    end

### OBJECT

@cdef :_PyObject_New PyPtr (PyPtr,)
@cdef :PyObject_ClearWeakRefs Cvoid (PyPtr,)
@cdef :PyObject_HasAttrString Cint (PyPtr, Cstring)
@cdef :PyObject_HasAttr Cint (PyPtr, PyPtr)
@cdef :PyObject_GetAttrString PyPtr (PyPtr, Cstring)
@cdef :PyObject_GetAttr PyPtr (PyPtr, PyPtr)
@cdef :PyObject_GenericGetAttr PyPtr (PyPtr, PyPtr)
@cdef :PyObject_SetAttrString Cint (PyPtr, Cstring, PyPtr)
@cdef :PyObject_SetAttr Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_GenericSetAttr Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_DelAttrString Cint (PyPtr, Cstring)
@cdef :PyObject_DelAttr Cint (PyPtr, PyPtr)
@cdef :PyObject_RichCompare PyPtr (PyPtr, PyPtr, Cint)
@cdef :PyObject_RichCompareBool Cint (PyPtr, PyPtr, Cint)
@cdef :PyObject_Repr PyPtr (PyPtr,)
@cdef :PyObject_ASCII PyPtr (PyPtr,)
@cdef :PyObject_Str PyPtr (PyPtr,)
@cdef :PyObject_Bytes PyPtr (PyPtr,)
@cdef :PyObject_IsSubclass Cint (PyPtr, PyPtr)
@cdef :PyObject_IsInstance Cint (PyPtr, PyPtr)
@cdef :PyObject_Hash Py_hash_t (PyPtr,)
@cdef :PyObject_IsTrue Cint (PyPtr,)
@cdef :PyObject_Length Py_ssize_t (PyPtr,)
@cdef :PyObject_GetItem PyPtr (PyPtr, PyPtr)
@cdef :PyObject_SetItem Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_DelItem Cint (PyPtr, PyPtr)
@cdef :PyObject_Dir PyPtr (PyPtr,)
@cdef :PyObject_GetIter PyPtr (PyPtr,)
@cdef :PyObject_Call PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :PyObject_CallObject PyPtr (PyPtr, PyPtr)

function PyObject_ReprAs(o, ::Type{T}) where {T<:Union{Vector{UInt8},Vector{Int8},String}}
    x = PyObject_Repr(o)
    isnull(x) && return PYERR()
    r = PyUnicode_As(x, T)
    Py_DecRef(x)
    r
end

function PyObject_StrAs(o, ::Type{T}) where {T<:Union{Vector{UInt8},Vector{Int8},String}}
    x = PyObject_Str(o)
    isnull(x) && return PYERR()
    r = PyUnicode_As(x, T)
    Py_DecRef(x)
    r
end

function PyObject_ASCIIAs(o, ::Type{T}) where {T<:Union{Vector{UInt8},Vector{Int8},String}}
    x = PyObject_ASCII(o)
    isnull(x) && return PYERR()
    r = PyUnicode_As(x, T)
    Py_DecRef(x)
    r
end

function PyObject_BytesAs(o, ::Type{T}) where {T<:Union{Vector{UInt8},Vector{Int8},String}}
    x = PyObject_Bytes(o)
    isnull(x) && return PYERR()
    r = PyBytes_As(x, T)
    Py_DecRef(x)
    r
end

PyObject_From(x::PyObjectRef) = (Py_IncRef(x.ptr); x.ptr)
PyObject_From(x::Nothing) = PyNone_New()
PyObject_From(x::Bool) = PyBool_From(x)
PyObject_From(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) = PyLong_From(x)
PyObject_From(x::Union{Float16,Float32,Float64}) = PyFloat_From(x)
PyObject_From(x::Union{String,SubString{String}}) = PyUnicode_From(x)
PyObject_From(x::Tuple) = PyTuple_From(x)
function PyObject_From(x)
    PyErr_SetString(PyExc_TypeError(), "Cannot convert this Julia '$(typeof(x))' to a Python object.")
    PyPtr()
end

function PyObject_CallArgs(f, args, kwargs=())
    if !isempty(kwargs)
        error("kwargs not implemented")
    elseif !isempty(args)
        argso = PyTuple_From(Tuple(args))
        isnull(argso) && return PyPtr()
        r = PyObject_CallObject(f, argso)
        Py_DecRef(argso)
        return r
    else
        PyObject_CallObject(f, C_NULL)
    end
end

PyObject_CallNice(f, args...; kwargs...) = PyObject_CallArgs(f, args, kwargs)

const PYOBJECT_AS_RULES = Dict{PyPtr, Union{Nothing,Function}}()
const PYOBJECT_AS_CRULES = IdDict{Type, Dict{PyPtr, Ptr{Cvoid}}}()
const PYOBJECT_AS_CRULES_CACHE = IdDict{Type, Dict{PyPtr, Any}}()

function PyObject_As__rule(t::PyPtr)
    name = Py_DecRef(PyObject_GetAttrString(t, "__name__")) do tnameo
        Py_DecRef(PyObject_GetAttrString(t, "__module__")) do mnameo
            tname = PyUnicode_As(tnameo, String)
            tname === PYERR() && return PYERR()
            mname = PyUnicode_As(mnameo, String)
            mname === PYERR() && return PYERR()
            "$mname.$tname"
        end
    end
    name == PYERR() && return PYERR()
    PyObject_As__rule(t, Val(Symbol(name)))
end

PyObject_As__rule(t::PyPtr, ::Val) = nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.NoneType")}) = Py_Is(t, Py_Type(Py_None())) ? PyNone_As : nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.bool")}) = Py_Is(t, PyBool_Type()) ? PyBool_As : nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.str")}) = Py_Is(t, PyUnicode_Type()) ? PyUnicode_As : nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.bytes")}) = Py_Is(t, PyBytes_Type()) ? PyBytes_As : nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.int")}) = Py_Is(t, PyLong_Type()) ? PyLong_As : nothing
PyObject_As__rule(t::PyPtr, ::Val{Symbol("builtins.float")}) = Py_Is(t, PyFloat_Type()) ? PyFloat_As : nothing

struct PyObject_As__crule_struct{T,F}
    f :: F
end
function (r::PyObject_As__crule_struct{T,F})(o::PyPtr, ref::Base.RefValue{Any}) where {T,F}
    res = r.f(o, T)
    if res === PYERR()
        return Cint(2)
    elseif res === NOTIMPLEMENTED()
        return Cint(1)
    else
        ref[] = res
        return Cint(0)
    end
end

function PyObject_As(o::PyPtr, ::Type{T}) where {T}
    # Run through the MRO, applying conversion rules that depend on the supertypes of the type of o.
    # For speed, the conversion functions are cached as C function pointers.
    # These take as inputs the PyPtr o and a Ptr{Any} in which to store the result.
    # They return 0 on success, 1 on no-conversion and 2 on error.
    mro = PyType_MRO(Py_Type(o))
    ref = Ref{Any}()
    rules = get!(Dict{PyPtr, Ptr{Cvoid}}, PYOBJECT_AS_CRULES, T)::Dict{PyPtr,Ptr{Cvoid}}
    for i in 1:PyTuple_Size(mro)
        t = PyTuple_GetItem(mro, i-1)
        isnull(t) && return PYERR()
        crule = get(rules, t, missing)
        if crule === missing
            rule = PyObject_As__rule(t)
            rule === PYERR() && return PYERR()
            if rule === nothing
                rules[t] = C_NULL
                continue
            else
                crulefunc = @cfunction($(PyObject_As__crule_struct{T, typeof(rule)}(rule)), Cint, (PyPtr, Any))
                get!(Dict{PyPtr, Any}, PYOBJECT_AS_CRULES_CACHE, T)[t] = crulefunc
                crule = rules[t] = Base.unsafe_convert(Ptr{Cvoid}, crulefunc)
            end
        elseif crule == C_NULL
            continue
        end
        res = ccall(crule, Cint, (PyPtr, Any), o, ref)
        if res == 0
            return ref[]::T
        elseif res == 2
            return PYERR()
        end
    end
    NOTIMPLEMENTED()
end
PyObject_As(o, ::Type{T}) where {T} = GC.@preserve o PyObject_As(Base.unsafe_convert(PyPtr, o), T)

### SEQUENCE

@cdef :PySequence_Length Py_ssize_t (PyPtr,)
@cdef :PySequence_GetItem PyPtr (PyPtr, Py_ssize_t)
@cdef :PySequence_Contains Cint (PyPtr, PyPtr)

### MAPPING

@cdef :PyMapping_HasKeyString Cint (PyPtr, Cstring)
@cdef :PyMapping_SetItemString Cint (PyPtr, Cstring, PyPtr)
@cdef :PyMapping_GetItemString PyPtr (PyPtr, Cstring)

PyMapping_ExtractOneAs(o, k, ::Type{T}) where {T} =
    Py_DecRef(PyMapping_GetItemString(o, string(k))) do x
        v = PyObject_As(x, T)
        if v === NOTIMPLEMENTED()
            PyErr_SetString(PyExc_TypeError(), "Cannot convert this '$(PyType_Name(Py_Type(x)))' at key '$k' to a Julia '$T'")
            PYERR()
        else
            v
        end
    end

PyMapping_ExtractAs(o::PyPtr, ::Type{NamedTuple{names,types}}) where {names, types} = begin
    t = PyMapping_ExtractAs(o, names, types)
    t === PYERR() ? PYERR() : NamedTuple{names,types}(t)
end
PyMapping_ExtractAs(o::PyPtr, names::Tuple, ::Type{types}) where {types<:Tuple} = begin
    v = PyMapping_ExtractOneAs(o, first(names), Base.tuple_type_head(types))
    v === PYERR() && return PYERR()
    vs = PyMapping_ExtractAs(o::PyPtr, Base.tail(names), Base.tuple_type_tail(types))
    vs === PYERR() && return PYERR()
    (v, vs...)
end
PyMapping_ExtractAs(o::PyPtr, names::Tuple{}, ::Type{Tuple{}}) = ()
PyMapping_ExtractAs(o, ::Type{T}) where {T} = GC.@preserve o PyMapping_ExtractAs(Base.unsafe_convert(PyPtr, o), T)

### METHOD

@cdef :PyInstanceMethod_New PyPtr (PyPtr,)

### STR

@cdef :PyUnicode_DecodeUTF8 PyPtr (Ptr{Cchar}, Py_ssize_t, Ptr{Cvoid})
@cdef :PyUnicode_AsUTF8String PyPtr (PyPtr,)

const PyUnicode_Type__ref = Ref(C_NULL)
PyUnicode_Type() = PyPtr(pyglobal(PyUnicode_Type__ref, :PyUnicode_Type))

PyUnicode_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_UNICODE_SUBCLASS)
PyUnicode_CheckExact(o) = Py_TypeCheckExact(o, PyUnicode_Type())

PyUnicode_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyUnicode_DecodeUTF8(pointer(s), sizeof(s), C_NULL)

PyUnicode_As(o, ::Type{T}) where {T} = begin
    if (S = _typeintersect(T, AbstractString)) != Union{}
        r = Py_DecRef(PyUnicode_AsUTF8String(o)) do b
            PyBytes_As(b, S)
        end
        r === NOTIMPLEMENTED() || return r
    end
    if Symbol <: T
        s = PyUnicode_As(o, String)
        s === PYERR() && return PYERR()
        s isa String && return Symbol(s)
    end
    if (S = _typeintersect(T, AbstractChar)) != Union{}
        s = PyUnicode_As(o, String)
        s === PYERR() && return PYERR()
        if s isa String
            if length(s) == 1
                c = first(s)
                r = tryconvert(S, c)
                r === NOTIMPLEMENTED() || return r
            end
        end
    end
    if (S = _typeintersect(T, AbstractVector)) != Union{}
        r = Py_DecRef(PyUnicode_AsUTF8String(o)) do b
            PyBytes_As(b, S)
        end
        r === NOTIMPLEMENTED() || return r
    end
    NOTIMPLEMENTED()
end

### BYTES

@cdef :PyBytes_FromStringAndSize PyPtr (Ptr{Cchar}, Py_ssize_t)
@cdef :PyBytes_AsStringAndSize Cint (PyPtr, Ptr{Ptr{Cchar}}, Ptr{Py_ssize_t})

PyBytes_From(s::Union{Vector{Cuchar},Vector{Cchar},String,SubString{String}}) =
    PyBytes_FromStringAndSize(pointer(s), sizeof(s))

PyBytes_As(o, ::Type{T}) where {T} = begin
    if (S = _typeintersect(T, AbstractVector{UInt8})) != Union{}
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyBytes_AsStringAndSize(o, ptr, len)
        ism1(err) && return PYERR()
        v = copy(Base.unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(ptr[]), len[]))
        r = tryconvert(S, v)
        r === NOTIMPLEMENTED() || return r
    end
    if (S = _typeintersect(T, AbstractVector{Int8})) != Union{}
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyBytes_AsStringAndSize(o, ptr, len)
        ism1(err) && return PYERR()
        v = copy(Base.unsafe_wrap(Vector{Int8}, Ptr{Int8}(ptr[]), len[]))
        r = tryconvert(S, v)
        r === NOTIMPLEMENTED() || return r
    end
    if (S = _typeintersect(T, AbstractString)) != Union{}
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyBytes_AsStringAndSize(o, ptr, len)
        ism1(err) && return PYERR()
        s = Base.unsafe_string(ptr[], len[])
        r = tryconvert(S, s)
        r === NOTIMPLEMENTED() || return r
    end
    NOTIMPLEMENTED()
end

### TUPLE

@cdef :PyTuple_New PyPtr (Py_ssize_t,)
@cdef :PyTuple_Size Py_ssize_t (PyPtr,)
@cdef :PyTuple_GetItem PyPtr (PyPtr, Py_ssize_t)
@cdef :PyTuple_SetItem Cint (PyPtr, Py_ssize_t, PyPtr)

function PyTuple_From(xs::Tuple)
    t = PyTuple_New(length(xs))
    isnull(t) && return PyPtr()
    for (i,x) in enumerate(xs)
        xo = PyObject_From(x)
        isnull(xo) && (Py_DecRef(t); return PyPtr())
        err = PyTuple_SetItem(t, i-1, xo) # steals xo
        ism1(err) && (Py_DecRef(t); return PyPtr())
    end
    return t
end

### TYPE

@cdef :PyType_IsSubtype Cint (PyPtr, PyPtr)
@cdef :PyType_Ready Cint (PyPtr,)

Py_Type(o) = GC.@preserve o UnsafePtr(Base.unsafe_convert(PyPtr, o)).type[!]
Py_TypeCheck(o, t) = PyType_IsSubtype(Py_Type(o), t)
Py_TypeCheckExact(o, t) = Py_Type(o) == Base.unsafe_convert(PyPtr, t)
Py_TypeCheckFast(o, f) = PyType_IsSubtypeFast(Py_Type(o), f)

PyType_Flags(o) = GC.@preserve o UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).flags[]
PyType_Name(o) = GC.@preserve o unsafe_string(UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).name[!])
PyType_MRO(o) = GC.@preserve o UnsafePtr{PyTypeObject}(Base.unsafe_convert(PyPtr, o)).mro[!]

PyType_IsSubtypeFast(s, f) = PyType_HasFeature(s, f)
PyType_HasFeature(s, f) = !iszero(PyType_Flags(s) & f)

const PyType_Type__ref = Ref(C_NULL)
PyType_Type() = PyPtr(pyglobal(PyType_Type__ref, :PyType_Type))

PyType_Check(o) = Py_TypeCheck(o, Py_TPFLAGS_TYPE_SUBCLASS)

PyType_CheckExact(o) = Py_TypeCheckExact(o, PyType_Type())

### NUMBER

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

### ITER

@cdef :PyIter_Next PyPtr (PyPtr,)

### BOOL

const PyBool_Type__ref = Ref(C_NULL)
PyBool_Type() = PyPtr(pyglobal(PyBool_Type__ref, :PyBool_Type))

const Py_True__ref = Ref(C_NULL)
Py_True() = PyPtr(pyglobal(Py_True__ref, :_Py_TrueStruct))

const Py_False__ref = Ref(C_NULL)
Py_False() = PyPtr(pyglobal(Py_False__ref, :_Py_FalseStruct))

PyBool_From(x::Bool) = (o = x ? Py_True() : Py_False(); Py_IncRef(o); o)

PyBool_Check(o) = Py_TypeCheckExact(o, PyBool_Type())

PyBool_As(o, ::Type{T}) where {T} =
    if Bool <: T
        if Py_Is(o, Py_True())
            true
        elseif Py_Is(o, Py_False())
            false
        else
            PyErr_SetString(PyExc_TypeError(), "not a 'bool'")
            PYERR()
        end
    else
        NOTIMPLEMENTED()
    end


### INT

@cdef :PyLong_FromLongLong PyPtr (Clonglong,)
@cdef :PyLong_FromUnsignedLongLong PyPtr (Culonglong,)
@cdef :PyLong_FromString PyPtr (Cstring, Ptr{Cvoid}, Cint)
@cdef :PyLong_AsLongLong Clonglong (PyPtr,)
@cdef :PyLong_AsUnsignedLongLong Culonglong (PyPtr,)

const PyLong_Type__ref = Ref(C_NULL)
PyLong_Type() = PyPtr(pyglobal(PyLong_Type__ref, :PyLong_Type))

PyLong_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_LONG_SUBCLASS)

PyLong_CheckExact(o) = Py_TypeCheckExact(o, PyLong_Type())

PyLong_From(x::Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128,BigInt}) =
    if x isa Signed && typemin(Clonglong) ≤ x ≤ typemax(Clonglong)
        PyLong_FromLongLong(x)
    elseif typemin(Culonglong) ≤ x ≤ typemax(Culonglong)
        PyLong_FromUnsignedLongLong(x)
    else
        PyLong_FromString(string(x), C_NULL, 10)
    end

PyLong_From(x::Integer) = begin
    y = tryconvert(BigInt, x)
    y === PYERR() && return PyPtr()
    y === NOTIMPLEMENTED() && (PyErr_SetString(PyExc_NotImplementedError(), "Cannot convert this Julia '$(typeof(x))' to a Python 'int'"); return PyPtr())
    PyLong_From(y::BigInt)
end

PyLong_As(o, ::Type{T}) where {T} = begin
    if (S = _typeintersect(T, Integer)) != Union{}
        # first try to convert to Clonglong (or Culonglong if unsigned)
        x = S <: Unsigned ? PyLong_AsUnsignedLongLong(o) : PyLong_AsLongLong(o)
        if !ism1(x) || !PyErr_IsSet()
            # success
            return tryconvert(S, x)
        elseif PyErr_IsSet(PyExc_OverflowError())
            # overflows Clonglong or Culonglong
            PyErr_Clear()
            if S in (Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128) && typemin(typeof(x)) ≤ typemin(S) && typemax(S) ≤ typemax(typeof(x))
                # definitely overflows S, give up now
                return NOTIMPLEMENTED()
            else
                # try converting to String then BigInt then S
                s = PyObject_StrAs(o, String)
                s === PYERR() && return PYERR()
                y = tryparse(BigInt, s)
                y === nothing && (PyErr_SetString(PyExc_ValueError(), "Cannot convert this '$(PyType_Name(Py_Type(o)))' to a Julia 'BigInt' because its string representation cannot be parsed as an integer"); return PYERR())
                return tryconvert(S, y::BigInt)
            end
        else
            # other error
            return PYERR()
        end
    elseif (S = _typeintersect(T, Real)) != Union{}
        return tryconvert(S, PyLong_As(o, Integer))
    elseif (S = _typeintersect(T, Number)) != Union{}
        return tryconvert(S, PyLong_As(o, Integer))
    else
        return tryconvert(T, PyLong_As(o, Integer))
    end
    NOTIMPLEMENTED()
end

### FLOAT

@cdef :PyFloat_FromDouble PyPtr (Cdouble,)
@cdef :PyFloat_AsDouble Cdouble (PyPtr,)

const PyFloat_Type__ref = Ref(C_NULL)
PyFloat_Type() = PyPtr(pyglobal(PyFloat_Type__ref, :PyFloat_Type))

PyFloat_Check(o) = Py_TypeCheck(o, PyFloat_Type())

PyFloat_CheckExact(o) = Py_TypeCheckExact(o, PyFloat_Type())

PyFloat_From(o::Union{Float16,Float32,Float64}) = PyFloat_FromDouble(o)

PyFloat_As(o, ::Type{T}) where {T} = begin
    x = PyFloat_AsDouble(o)
    ism1(x) && PyErr_IsSet() && return PYERR()
    if Float64 <: T
        return convert(Float64, x)
    elseif Float32 <: T
        return convert(Float32, x)
    elseif Float16 <: T
        return convert(Float16, x)
    elseif (S = _typeintersect(T, AbstractFloat)) != Union{}
        return tryconvert(S, x)
    elseif (S = _typeintersect(T, Real)) != Union{}
        return tryconvert(S, x)
    elseif (S = _typeintersect(T, Number)) != Union{}
        return tryconvert(S, x)
    else
        return tryconvert(T, x)
    end
end

### COMPLEX

@cdef :PyComplex_RealAsDouble Cdouble (PyPtr,)
@cdef :PyComplex_ImagAsDouble Cdouble (PyPtr,)

### LIST

@cdef :PyList_New PyPtr (Py_ssize_t,)
@cdef :PyList_Append Cint (PyPtr, PyPtr)
@cdef :PyList_AsTuple PyPtr (PyPtr,)

### DICT

@cdef :PyDict_New PyPtr ()
@cdef :PyDict_SetItem Cint (PyPtr, PyPtr, PyPtr)
@cdef :PyDict_SetItemString Cint (PyPtr, Cstring, PyPtr)
@cdef :PyDict_DelItemString Cint (PyPtr, Cstring)

### BUFFER

function PyObject_CheckBuffer(o)
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    !isnull(p) && !isnull(p.get[])
end

function PyObject_GetBuffer(o, b, flags)
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if isnull(p) || isnull(p.get[])
        PyErr_SetString(unsafe_load(Ptr{PyPtr}(pyglobal(:PyExc_TypeError))), "a bytes-like object is required, not '$(String(UnsafePtr{PyTypeObject}(Py_Type(o)).name[]))'")
        return Cint(-1)
    end
    ccall(p.get[!], Cint, (PyPtr, Ptr{Py_buffer}, Cint), o, b, flags)
end

function PyBuffer_Release(_b)
    b = UnsafePtr(Base.unsafe_convert(Ptr{Py_buffer}, _b))
    o = b.obj[]
    isnull(o) && return
    p = UnsafePtr{PyTypeObject}(Py_Type(o)).as_buffer[]
    if (!isnull(p) && !isnull(p.release[]))
        ccall(p.release[!], Cvoid, (PyPtr, Ptr{Py_buffer}), o, b)
    end
    b.obj[] = C_NULL
    Py_DecRef(o)
    return
end

### INPUT HOOK

function PyOS_RunInputHook()
    hook = unsafe_load(Ptr{Ptr{Cvoid}}(pyglobal(:PyOS_InputHook)))
    isnull(hook) || ccall(hook, Cint, ())
    nothing
end

end
