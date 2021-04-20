const CAPI_FUNCS = [
    # INITIALIZE
    :Py_Initialize,
    :Py_InitializeEx,
    :Py_Finalize,
    :Py_FinalizeEx,
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
]

const CAPI_EXCEPTIONS = [
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
]

@eval @kwdef mutable struct CAPIPointers
    $([:($name :: Ptr{Cvoid} = C_NULL) for name in CAPI_FUNCS]...)
    $([:($name :: PyPtr = C_NULL) for name in CAPI_EXCEPTIONS]...)
end

const POINTERS = CAPIPointers()

@eval init!(p::CAPIPointers) = begin
    $([:(p.$name = pyglobal($(QuoteNode(name)))) for name in CAPI_FUNCS]...)
    $([:(p.$name = pyloadglobal(PyPtr, $(QuoteNode(name)))) for name in CAPI_EXCEPTIONS]...)
end

init_pointers() = init!(POINTERS)
