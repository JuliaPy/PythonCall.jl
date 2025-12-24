const CAPI_FUNC_SIGS = Dict{Symbol,Pair{Tuple,Type}}(
    # INITIALIZE
    :Py_Initialize => () => Cvoid,
    :Py_InitializeEx => (Cint,) => Cvoid,
    :Py_FinalizeEx => () => Cint,
    :Py_AtExit => (Ptr{Cvoid},) => Cint,
    :Py_IsInitialized => () => Cint,
    :Py_SetPythonHome => (Ptr{Cwchar_t},) => Cvoid,
    :Py_SetProgramName => (Ptr{Cwchar_t},) => Cvoid,
    :Py_GetVersion => () => Ptr{Cchar},
    # REFCOUNT
    :Py_IncRef => (PyPtr,) => Cvoid,
    :Py_DecRef => (PyPtr,) => Cvoid,
    # EVAL
    :PyEval_EvalCode => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :Py_CompileString => (Ptr{Cchar}, Ptr{Cchar}, Cint) => PyPtr,
    :PyEval_GetBuiltins => () => PyPtr,
    :PyRun_InteractiveOne => (Ptr{Cvoid}, Ptr{Cchar}) => Cint,
    # GIL & THREADS
    :PyEval_SaveThread => () => Ptr{Cvoid},
    :PyEval_RestoreThread => (Ptr{Cvoid},) => Cvoid,
    :PyGILState_Ensure => () => PyGILState_STATE,
    :PyGILState_Release => (PyGILState_STATE,) => Cvoid,
    :PyGILState_GetThisThreadState => () => Ptr{Cvoid},
    :PyGILState_Check => () => Cint,
    # IMPORT
    :PyImport_ImportModule => (Ptr{Cchar},) => PyPtr,
    :PyImport_Import => (PyPtr,) => PyPtr,
    :PyImport_ImportModuleLevelObject => (PyPtr, PyPtr, PyPtr, PyPtr, Cint) => PyPtr,
    :PyImport_GetModuleDict => () => PyPtr, # borrowed
    # MODULE
    :PyModule_GetDict => (PyPtr,) => PyPtr, # borrowed
    # ERRORS
    :PyErr_Print => () => Cvoid,
    :PyErr_Occurred => () => PyPtr, # borrowed
    :PyErr_ExceptionMatches => (PyPtr,) => Cint,
    :PyErr_GivenExceptionMatches => (PyPtr, PyPtr) => Cint,
    :PyErr_Clear => () => Cvoid,
    :PyErr_SetNone => (PyPtr,) => Cvoid,
    :PyErr_SetString => (PyPtr, Ptr{Cchar}) => Cvoid,
    :PyErr_SetObject => (PyPtr, PyPtr) => Cvoid,
    :PyErr_Fetch => (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr}) => Cvoid,
    :PyErr_NormalizeException => (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr}) => Cvoid,
    :PyErr_Restore => (PyPtr, PyPtr, PyPtr) => Cvoid, # steals
    # OBJECT
    :_PyObject_New => (PyPtr,) => PyPtr,
    :PyObject_ClearWeakRefs => (PyPtr,) => Cvoid,
    :PyObject_HasAttrString => (PyPtr, Ptr{Cchar}) => Cint,
    :PyObject_HasAttr => (PyPtr, PyPtr) => Cint,
    :PyObject_GetAttrString => (PyPtr, Ptr{Cchar}) => PyPtr,
    :PyObject_GetAttr => (PyPtr, PyPtr) => PyPtr,
    :PyObject_GenericGetAttr => (PyPtr, PyPtr) => PyPtr,
    :PyObject_SetAttrString => (PyPtr, Ptr{Cchar}, PyPtr) => Cint,
    :PyObject_SetAttr => (PyPtr, PyPtr, PyPtr) => Cint,
    :PyObject_GenericSetAttr => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :PyObject_RichCompare => (PyPtr, PyPtr, Cint) => PyPtr,
    :PyObject_RichCompareBool => (PyPtr, PyPtr, Cint) => Cint,
    :PyObject_Repr => (PyPtr,) => PyPtr,
    :PyObject_ASCII => (PyPtr,) => PyPtr,
    :PyObject_Str => (PyPtr,) => PyPtr,
    :PyObject_Bytes => (PyPtr,) => PyPtr,
    :PyObject_IsSubclass => (PyPtr, PyPtr) => Cint,
    :PyObject_IsInstance => (PyPtr, PyPtr) => Cint,
    :PyObject_Hash => (PyPtr,) => Py_hash_t,
    :PyObject_IsTrue => (PyPtr,) => Cint,
    :PyObject_Not => (PyPtr,) => Cint,
    :PyObject_Length => (PyPtr,) => Py_ssize_t,
    :PyObject_GetItem => (PyPtr, PyPtr) => PyPtr,
    :PyObject_SetItem => (PyPtr, PyPtr, PyPtr) => Cint,
    :PyObject_DelItem => (PyPtr, PyPtr) => Cint,
    :PyObject_Dir => (PyPtr,) => PyPtr,
    :PyObject_GetIter => (PyPtr,) => PyPtr,
    :PyObject_Call => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :PyObject_CallObject => (PyPtr, PyPtr) => PyPtr,
    # TYPE
    :PyType_IsSubtype => (PyPtr, PyPtr) => Cint,
    :PyType_Ready => (PyPtr,) => Cint,
    :PyType_GenericNew => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :PyType_FromSpec => (Ptr{Cvoid},) => PyPtr,
    # MAPPING
    :PyMapping_HasKeyString => (PyPtr, Ptr{Cchar}) => Cint,
    :PyMapping_SetItemString => (PyPtr, Ptr{Cchar}, PyPtr) => Cint,
    :PyMapping_GetItemString => (PyPtr, Ptr{Cchar}) => PyPtr,
    # SEQUENCE
    :PySequence_Length => (PyPtr,) => Py_ssize_t,
    :PySequence_GetItem => (PyPtr, Py_ssize_t) => PyPtr,
    :PySequence_SetItem => (PyPtr, Py_ssize_t, PyPtr) => Cint,
    :PySequence_Contains => (PyPtr, PyPtr) => Cint,
    # NUMBER
    :PyNumber_Add => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Subtract => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Multiply => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_MatrixMultiply => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_FloorDivide => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_TrueDivide => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Remainder => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Divmod => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Power => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :PyNumber_Negative => (PyPtr,) => PyPtr,
    :PyNumber_Positive => (PyPtr,) => PyPtr,
    :PyNumber_Absolute => (PyPtr,) => PyPtr,
    :PyNumber_Invert => (PyPtr,) => PyPtr,
    :PyNumber_Lshift => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Rshift => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_And => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Xor => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Or => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceAdd => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceSubtract => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceMultiply => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceMatrixMultiply => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceFloorDivide => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceTrueDivide => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceRemainder => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlacePower => (PyPtr, PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceLshift => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceRshift => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceAnd => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceXor => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_InPlaceOr => (PyPtr, PyPtr) => PyPtr,
    :PyNumber_Long => (PyPtr,) => PyPtr,
    :PyNumber_Float => (PyPtr,) => PyPtr,
    :PyNumber_Index => (PyPtr,) => PyPtr,
    # ITERATION
    :PyIter_Next => (PyPtr,) => PyPtr,
    # INT
    :PyLong_FromLongLong => (Clonglong,) => PyPtr,
    :PyLong_FromUnsignedLongLong => (Culonglong,) => PyPtr,
    :PyLong_FromString => (Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint) => PyPtr,
    :PyLong_AsLongLong => (PyPtr,) => Clonglong,
    :PyLong_AsUnsignedLongLong => (PyPtr,) => Culonglong,
    # FLOAT
    :PyFloat_FromDouble => (Cdouble,) => PyPtr,
    :PyFloat_AsDouble => (PyPtr,) => Cdouble,
    # COMPLEX
    :PyComplex_FromDoubles => (Cdouble, Cdouble) => PyPtr,
    :PyComplex_RealAsDouble => (PyPtr,) => Cdouble,
    :PyComplex_ImagAsDouble => (PyPtr,) => Cdouble,
    :PyComplex_AsCComplex => (PyPtr,) => Py_complex,
    # STR
    :PyUnicode_DecodeUTF8 => (Ptr{Cchar}, Py_ssize_t, Ptr{Cchar}) => PyPtr,
    :PyUnicode_AsUTF8AndSize => (PyPtr, Ptr{Py_ssize_t}) => Ptr{Cchar},
    :PyUnicode_AsUTF8String => (PyPtr,) => PyPtr,
    :PyUnicode_InternInPlace => (Ptr{PyPtr},) => Cvoid,
    # BYTES
    :PyBytes_FromStringAndSize => (Ptr{Cchar}, Py_ssize_t) => PyPtr,
    :PyBytes_AsStringAndSize => (PyPtr, Ptr{Ptr{Cchar}}, Ptr{Py_ssize_t}) => Cint,
    # TUPLE
    :PyTuple_New => (Py_ssize_t,) => PyPtr,
    :PyTuple_Size => (PyPtr,) => Py_ssize_t,
    :PyTuple_GetItem => (PyPtr, Py_ssize_t) => PyPtr, # borrowed
    :PyTuple_SetItem => (PyPtr, Py_ssize_t, PyPtr) => Cint, # steals
    # LIST
    :PyList_New => (Py_ssize_t,) => PyPtr,
    :PyList_Append => (PyPtr, PyPtr) => Cint,
    :PyList_AsTuple => (PyPtr,) => PyPtr,
    :PyList_SetItem => (PyPtr, Py_ssize_t, PyPtr) => Cint, # steals
    # DICT
    :PyDict_New => () => PyPtr,
    :PyDict_GetItem => (PyPtr, PyPtr) => PyPtr, # borrowed
    :PyDict_GetItemString => (PyPtr, Ptr{Cchar}) => PyPtr, # borrowed
    :PyDict_SetItem => (PyPtr, PyPtr, PyPtr) => Cint,
    :PyDict_SetItemString => (PyPtr, Ptr{Cchar}, PyPtr) => Cint,
    :PyDict_DelItemString => (PyPtr, Ptr{Cchar}) => Cint,
    # SET
    :PySet_New => (PyPtr,) => PyPtr,
    :PyFrozenSet_New => (PyPtr,) => PyPtr,
    :PySet_Add => (PyPtr, PyPtr) => Cint,
    # SLICE
    :PySlice_New => (PyPtr, PyPtr, PyPtr) => PyPtr,
    # METHOD
    :PyInstanceMethod_New => (PyPtr,) => PyPtr,
    # CAPSULE
    :PyCapsule_New => (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}) => PyPtr,
    :PyCapsule_GetName => (PyPtr,) => Ptr{Cchar},
    :PyCapsule_SetName => (PyPtr, Ptr{Cchar}) => Cint,
    :PyCapsule_GetPointer => (PyPtr, Ptr{Cchar}) => Ptr{Cvoid},
    :PyCapsule_SetDestructor => (PyPtr, Ptr{Cvoid}) => Cint,
)

const CAPI_EXCEPTIONS = Set([
    :PyExc_BaseException,
    :PyExc_Exception,
    :PyExc_StopIteration,
    :PyExc_GeneratorExit,
    :PyExc_ArithmeticError,
    :PyExc_LookupError,
    :PyExc_AssertionError,
    :PyExc_AttributeError,
    :PyExc_BufferError,
    :PyExc_EOFError,
    :PyExc_FloatingPointError,
    :PyExc_OSError,
    :PyExc_ImportError,
    :PyExc_IndexError,
    :PyExc_KeyError,
    :PyExc_KeyboardInterrupt,
    :PyExc_MemoryError,
    :PyExc_NameError,
    :PyExc_OverflowError,
    :PyExc_RuntimeError,
    :PyExc_RecursionError,
    :PyExc_NotImplementedError,
    :PyExc_SyntaxError,
    :PyExc_IndentationError,
    :PyExc_TabError,
    :PyExc_ReferenceError,
    :PyExc_SystemError,
    :PyExc_SystemExit,
    :PyExc_TypeError,
    :PyExc_UnboundLocalError,
    :PyExc_UnicodeError,
    :PyExc_UnicodeEncodeError,
    :PyExc_UnicodeDecodeError,
    :PyExc_UnicodeTranslateError,
    :PyExc_ValueError,
    :PyExc_ZeroDivisionError,
    :PyExc_BlockingIOError,
    :PyExc_BrokenPipeError,
    :PyExc_ChildProcessError,
    :PyExc_ConnectionError,
    :PyExc_ConnectionAbortedError,
    :PyExc_ConnectionRefusedError,
    :PyExc_FileExistsError,
    :PyExc_FileNotFoundError,
    :PyExc_InterruptedError,
    :PyExc_IsADirectoryError,
    :PyExc_NotADirectoryError,
    :PyExc_PermissionError,
    :PyExc_ProcessLookupError,
    :PyExc_TimeoutError,
    :PyExc_EnvironmentError,
    :PyExc_IOError,
    # :PyExc_WindowsError, # only on Windows
    :PyExc_Warning,
    :PyExc_UserWarning,
    :PyExc_DeprecationWarning,
    :PyExc_PendingDeprecationWarning,
    :PyExc_SyntaxWarning,
    :PyExc_RuntimeWarning,
    :PyExc_FutureWarning,
    :PyExc_ImportWarning,
    :PyExc_UnicodeWarning,
    :PyExc_BytesWarning,
    :PyExc_ResourceWarning,
])

const CAPI_FUNCS = Set(keys(CAPI_FUNC_SIGS))

const CAPI_OBJECTS = Set([
    # TYPES
    :PyBool_Type,
    :PyBytes_Type,
    :PyComplex_Type,
    :PyDict_Type,
    :PyFloat_Type,
    :PyLong_Type,
    :PyList_Type,
    :PyBaseObject_Type,
    :PySet_Type,
    :PyFrozenSet_Type,
    :PySlice_Type,
    :PyUnicode_Type,
    :PyTuple_Type,
    :PyType_Type,
    :PyCapsule_Type,
    # OTHERS
    :_Py_TrueStruct,
    :_Py_FalseStruct,
    :_Py_NotImplementedStruct,
    :_Py_NoneStruct,
    :_Py_EllipsisObject,
])

@eval @kwdef mutable struct CAPIPointers
    $([:($name::Ptr{Cvoid} = C_NULL) for name in CAPI_FUNCS]...)
    $([:($name::PyPtr = C_NULL) for name in CAPI_EXCEPTIONS]...)
    $([:($name::PyPtr = C_NULL) for name in CAPI_OBJECTS]...)
    PyOS_InputHookPtr::Ptr{Ptr{Cvoid}} = C_NULL
end

const POINTERS = CAPIPointers()

@eval init_pointers(p::CAPIPointers = POINTERS, lib::Ptr = CTX.lib_ptr) = begin
    $([
        :(p.$name = dlsym(lib, $(QuoteNode(name))))
        for name in CAPI_FUNCS
    ]...)
    $(
        [
            :(p.$name =
                    Base.unsafe_load(Ptr{PyPtr}(dlsym(lib, $(QuoteNode(name)))::Ptr))) for name in CAPI_EXCEPTIONS
        ]...
    )
    $([:(p.$name = dlsym(lib, $(QuoteNode(name)))) for name in CAPI_OBJECTS]...)
    p.PyOS_InputHookPtr = dlsym(CTX.lib_ptr, :PyOS_InputHook)
end

for (name, (argtypes, rettype)) in CAPI_FUNC_SIGS
    args = [Symbol("x", i) for (i, _) in enumerate(argtypes)]
    @eval $name($(args...)) = ccall(POINTERS.$name, $rettype, ($(argtypes...),), $(args...))
end
