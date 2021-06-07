const CAPI_FUNCS = Set([
    # INITIALIZE
    :Py_Initialize,
    :Py_InitializeEx,
    :Py_Finalize,
    :Py_FinalizeEx, # Python 3.6+
    :Py_AtExit,
    :Py_IsInitialized,
    :Py_SetPythonHome,
    :Py_SetProgramName,
    :Py_GetVersion,
    # REFCOUNT
    :Py_IncRef,
    :Py_DecRef,
    # EVAL
    :PyEval_EvalCode,
    :Py_CompileString,
    :PyEval_GetBuiltins,
    :PyRun_InteractiveOne,
    # GIL & THREADS
    :PyEval_SaveThread,
    :PyEval_RestoreThread,
    :PyGILState_Ensure,
    :PyGILState_Release,
    # IMPORT
    :PyImport_ImportModule,
    :PyImport_Import,
    :PyImport_ImportModuleLevelObject,
    :PyImport_GetModuleDict,
    # MODULE
    :PyModule_GetDict,
    # ERRORS
    :PyErr_Occurred,
    :PyErr_GivenExceptionMatches,
    :PyErr_Clear,
    :PyErr_SetNone,
    :PyErr_SetString,
    :PyErr_SetObject,
    :PyErr_Fetch,
    :PyErr_NormalizeException,
    :PyErr_Restore,
    # OBJECT
    :_PyObject_New,
    :PyObject_ClearWeakRefs,
    :PyObject_HasAttrString,
    :PyObject_HasAttr,
    :PyObject_GetAttrString,
    :PyObject_GetAttr,
    :PyObject_GenericGetAttr,
    :PyObject_SetAttrString,
    :PyObject_SetAttr,
    :PyObject_GenericSetAttr,
    :PyObject_RichCompare,
    :PyObject_RichCompareBool,
    :PyObject_Repr,
    :PyObject_ASCII,
    :PyObject_Str,
    :PyObject_Bytes,
    :PyObject_IsSubclass,
    :PyObject_IsInstance,
    :PyObject_Hash,
    :PyObject_IsTrue,
    :PyObject_Not,
    :PyObject_Length,
    :PyObject_GetItem,
    :PyObject_SetItem,
    :PyObject_DelItem,
    :PyObject_Dir,
    :PyObject_GetIter,
    :PyObject_Call,
    :PyObject_CallObject,
    # TYPE
    :PyType_IsSubtype,
    :PyType_Ready,
    # MAPPING
    :PyMapping_HasKeyString,
    :PyMapping_SetItemString,
    :PyMapping_GetItemString,
    # SEQUENCE
    :PySequence_Length,
    :PySequence_GetItem,
    :PySequence_SetItem,
    :PySequence_Contains,
    # NUMBER
    :PyNumber_Add,
    :PyNumber_Subtract,
    :PyNumber_Multiply,
    :PyNumber_MatrixMultiply,
    :PyNumber_FloorDivide,
    :PyNumber_TrueDivide,
    :PyNumber_Remainder,
    :PyNumber_Divmod,
    :PyNumber_Power,
    :PyNumber_Negative,
    :PyNumber_Positive,
    :PyNumber_Absolute,
    :PyNumber_Invert,
    :PyNumber_Lshift,
    :PyNumber_Rshift,
    :PyNumber_And,
    :PyNumber_Xor,
    :PyNumber_Or,
    :PyNumber_InPlaceAdd,
    :PyNumber_InPlaceSubtract,
    :PyNumber_InPlaceMultiply,
    :PyNumber_InPlaceMatrixMultiply,
    :PyNumber_InPlaceFloorDivide,
    :PyNumber_InPlaceTrueDivide,
    :PyNumber_InPlaceRemainder,
    :PyNumber_InPlacePower,
    :PyNumber_InPlaceLshift,
    :PyNumber_InPlaceRshift,
    :PyNumber_InPlaceAnd,
    :PyNumber_InPlaceXor,
    :PyNumber_InPlaceOr,
    :PyNumber_Long,
    :PyNumber_Float,
    :PyNumber_Index,
    # ITERATION
    :PyIter_Next,
    # INT
    :PyLong_FromLongLong,
    :PyLong_FromUnsignedLongLong,
    :PyLong_FromString,
    :PyLong_AsLongLong,
    :PyLong_AsUnsignedLongLong,
    # FLOAT
    :PyFloat_FromDouble,
    :PyFloat_AsDouble,
    # COMPLEX
    :PyComplex_FromDoubles,
    :PyComplex_RealAsDouble,
    :PyComplex_ImagAsDouble,
    :PyComplex_AsCComplex,
    # STR
    :PyUnicode_DecodeUTF8,
    :PyUnicode_AsUTF8String,
    :PyUnicode_InternInPlace,
    # BYTES
    :PyBytes_FromStringAndSize,
    :PyBytes_AsStringAndSize,
    # TUPLE
    :PyTuple_New,
    :PyTuple_Size,
    :PyTuple_GetItem,
    :PyTuple_SetItem,
    # LIST
    :PyList_New,
    :PyList_Append,
    :PyList_AsTuple,
    :PyList_SetItem,
    # DICT
    :PyDict_New,
    :PyDict_GetItem,
    :PyDict_GetItemString,
    :PyDict_SetItem,
    :PyDict_SetItemString,
    :PyDict_DelItemString,
    # SET
    :PySet_New,
    :PyFrozenSet_New,
    :PySet_Add,
    # SLICE
    :PySlice_New,
    # METHOD
    :PyInstanceMethod_New,
])

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
    # OTHERS
    :_Py_TrueStruct,
    :_Py_FalseStruct,
    :_Py_NotImplementedStruct,
    :_Py_NoneStruct,
    :_Py_EllipsisObject,
])

const CAPI_OBJECTS_NOINIT = Set([
    :PyFraction_Type,
    :PyRange_Type,
])

@eval @kwdef mutable struct CAPIPointers
    $([:($name :: Ptr{Cvoid} = C_NULL) for name in CAPI_FUNCS]...)
    $([:($name :: PyPtr = C_NULL) for name in [CAPI_EXCEPTIONS; CAPI_OBJECTS; CAPI_OBJECTS_NOINIT]]...)
end

# const POINTERS = CAPIPointers()

# @eval init!(p::CAPIPointers) = begin
#     $([
#         if name == :Py_FinalizeEx
#             :(p.$name = dlsym_e(CONFIG.libptr, $(QuoteNode(name))))
#         else
#             :(p.$name = dlsym(CONFIG.libptr, $(QuoteNode(name))))
#         end
#         for name in CAPI_FUNCS
#     ]...)
#     $([:(p.$name = pyloadglobal(PyPtr, $(QuoteNode(name)))) for name in CAPI_EXCEPTIONS]...)
#     $([:(p.$name = pyglobal($(QuoteNode(name)))) for name in CAPI_OBJECTS]...)
# end

# init_pointers() = init!(POINTERS)
