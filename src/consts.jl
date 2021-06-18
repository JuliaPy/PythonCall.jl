# TYPES
const pybooltype = pynull(); export pybooltype;
const pybytestype = pynull(); export pybytestype;
const pycomplextype = pynull(); export pycomplextype;
const pydicttype = pynull(); export pydicttype;
const pyfloattype = pynull(); export pyfloattype;
const pyinttype = pynull(); export pyinttype;
const pylisttype = pynull(); export pylisttype;
const pyobjecttype = pynull(); export pyobjecttype;
const pysettype = pynull(); export pysettype;
const pyfrozensettype = pynull(); export pyfrozensettype;
const pyslicetype = pynull(); export pyslicetype;
const pystrtype = pynull(); export pystrtype;
const pytupletype = pynull(); export pytupletype;
const pytypetype = pynull(); export pytypetype;
# OTHERS
const pyTrue = pynull(); export pyTrue;
const pyFalse = pynull(); export pyFalse;
const pyNotImplemented = pynull(); export pyNotImplemented;
const pyNone = pynull(); export pyNone;
const pyEllipsis = pynull(); export pyEllipsis;
# EXCEPTIONS
const pyBaseException = pynull(); export pyBaseException;
const pyException = pynull(); export pyException;
const pyStopIteration = pynull(); export pyStopIteration;
const pyGeneratorExit = pynull(); export pyGeneratorExit;
const pyArithmeticError = pynull(); export pyArithmeticError;
const pyLookupError = pynull(); export pyLookupError;
const pyAssertionError = pynull(); export pyAssertionError;
const pyAttributeError = pynull(); export pyAttributeError;
const pyBufferError = pynull(); export pyBufferError;
const pyEOFError = pynull(); export pyEOFError;
const pyFloatingPointError = pynull(); export pyFloatingPointError;
const pyOSError = pynull(); export pyOSError;
const pyImportError = pynull(); export pyImportError;
const pyIndexError = pynull(); export pyIndexError;
const pyKeyError = pynull(); export pyKeyError;
const pyKeyboardInterrupt = pynull(); export pyKeyboardInterrupt;
const pyMemoryError = pynull(); export pyMemoryError;
const pyNameError = pynull(); export pyNameError;
const pyOverflowError = pynull(); export pyOverflowError;
const pyRuntimeError = pynull(); export pyRuntimeError;
const pyRecursionError = pynull(); export pyRecursionError;
const pyNotImplementedError = pynull(); export pyNotImplementedError;
const pySyntaxError = pynull(); export pySyntaxError;
const pyIndentationError = pynull(); export pyIndentationError;
const pyTabError = pynull(); export pyTabError;
const pyReferenceError = pynull(); export pyReferenceError;
const pySystemError = pynull(); export pySystemError;
const pySystemExit = pynull(); export pySystemExit;
const pyTypeError = pynull(); export pyTypeError;
const pyUnboundLocalError = pynull(); export pyUnboundLocalError;
const pyUnicodeError = pynull(); export pyUnicodeError;
const pyUnicodeEncodeError = pynull(); export pyUnicodeEncodeError;
const pyUnicodeDecodeError = pynull(); export pyUnicodeDecodeError;
const pyUnicodeTranslateError = pynull(); export pyUnicodeTranslateError;
const pyValueError = pynull(); export pyValueError;
const pyZeroDivisionError = pynull(); export pyZeroDivisionError;
const pyBlockingIOError = pynull(); export pyBlockingIOError;
const pyBrokenPipeError = pynull(); export pyBrokenPipeError;
const pyChildProcessError = pynull(); export pyChildProcessError;
const pyConnectionError = pynull(); export pyConnectionError;
const pyConnectionAbortedError = pynull(); export pyConnectionAbortedError;
const pyConnectionRefusedError = pynull(); export pyConnectionRefusedError;
const pyFileExistsError = pynull(); export pyFileExistsError;
const pyFileNotFoundError = pynull(); export pyFileNotFoundError;
const pyInterruptedError = pynull(); export pyInterruptedError;
const pyIsADirectoryError = pynull(); export pyIsADirectoryError;
const pyNotADirectoryError = pynull(); export pyNotADirectoryError;
const pyPermissionError = pynull(); export pyPermissionError;
const pyProcessLookupError = pynull(); export pyProcessLookupError;
const pyTimeoutError = pynull(); export pyTimeoutError;
const pyEnvironmentError = pynull(); export pyEnvironmentError;
const pyIOError = pynull(); export pyIOError;
const pyWarning = pynull(); export pyWarning;
const pyUserWarning = pynull(); export pyUserWarning;
const pyDeprecationWarning = pynull(); export pyDeprecationWarning;
const pyPendingDeprecationWarning = pynull(); export pyPendingDeprecationWarning;
const pySyntaxWarning = pynull(); export pySyntaxWarning;
const pyRuntimeWarning = pynull(); export pyRuntimeWarning;
const pyFutureWarning = pynull(); export pyFutureWarning;
const pyImportWarning = pynull(); export pyImportWarning;
const pyUnicodeWarning = pynull(); export pyUnicodeWarning;
const pyBytesWarning = pynull(); export pyBytesWarning;
const pyResourceWarning = pynull(); export pyResourceWarning;
# BUILTINS NOT AVAILABLE FROM C
const pyrangetype = pynull(); export pyrangetype;

function init_consts()
    # TYPES
    setnewptr!(pybooltype, C.POINTERS.PyBool_Type)
    setnewptr!(pybytestype, C.POINTERS.PyBytes_Type)
    setnewptr!(pycomplextype, C.POINTERS.PyComplex_Type)
    setnewptr!(pydicttype, C.POINTERS.PyDict_Type)
    setnewptr!(pyfloattype, C.POINTERS.PyFloat_Type)
    setnewptr!(pyinttype, C.POINTERS.PyLong_Type)
    setnewptr!(pylisttype, C.POINTERS.PyList_Type)
    setnewptr!(pyobjecttype, C.POINTERS.PyBaseObject_Type)
    setnewptr!(pysettype, C.POINTERS.PySet_Type)
    setnewptr!(pyfrozensettype, C.POINTERS.PyFrozenSet_Type)
    setnewptr!(pyslicetype, C.POINTERS.PySlice_Type)
    setnewptr!(pystrtype, C.POINTERS.PyUnicode_Type)
    setnewptr!(pytupletype, C.POINTERS.PyTuple_Type)
    setnewptr!(pytypetype, C.POINTERS.PyType_Type)
    # OTHERS
    setnewptr!(pyTrue, C.POINTERS._Py_TrueStruct)
    setnewptr!(pyFalse, C.POINTERS._Py_FalseStruct)
    setnewptr!(pyNotImplemented, C.POINTERS._Py_NotImplementedStruct)
    setnewptr!(pyNone, C.POINTERS._Py_NoneStruct)
    setnewptr!(pyEllipsis, C.POINTERS._Py_EllipsisObject)
    # EXCEPTIONS
    setnewptr!(pyBaseException, C.POINTERS.PyExc_BaseException)
    setnewptr!(pyException, C.POINTERS.PyExc_Exception)
    setnewptr!(pyStopIteration, C.POINTERS.PyExc_StopIteration)
    setnewptr!(pyGeneratorExit, C.POINTERS.PyExc_GeneratorExit)
    setnewptr!(pyArithmeticError, C.POINTERS.PyExc_ArithmeticError)
    setnewptr!(pyLookupError, C.POINTERS.PyExc_LookupError)
    setnewptr!(pyAssertionError, C.POINTERS.PyExc_AssertionError)
    setnewptr!(pyAttributeError, C.POINTERS.PyExc_AttributeError)
    setnewptr!(pyBufferError, C.POINTERS.PyExc_BufferError)
    setnewptr!(pyEOFError, C.POINTERS.PyExc_EOFError)
    setnewptr!(pyFloatingPointError, C.POINTERS.PyExc_FloatingPointError)
    setnewptr!(pyOSError, C.POINTERS.PyExc_OSError)
    setnewptr!(pyImportError, C.POINTERS.PyExc_ImportError)
    setnewptr!(pyIndexError, C.POINTERS.PyExc_IndexError)
    setnewptr!(pyKeyError, C.POINTERS.PyExc_KeyError)
    setnewptr!(pyKeyboardInterrupt, C.POINTERS.PyExc_KeyboardInterrupt)
    setnewptr!(pyMemoryError, C.POINTERS.PyExc_MemoryError)
    setnewptr!(pyNameError, C.POINTERS.PyExc_NameError)
    setnewptr!(pyOverflowError, C.POINTERS.PyExc_OverflowError)
    setnewptr!(pyRuntimeError, C.POINTERS.PyExc_RuntimeError)
    setnewptr!(pyRecursionError, C.POINTERS.PyExc_RecursionError)
    setnewptr!(pyNotImplementedError, C.POINTERS.PyExc_NotImplementedError)
    setnewptr!(pySyntaxError, C.POINTERS.PyExc_SyntaxError)
    setnewptr!(pyIndentationError, C.POINTERS.PyExc_IndentationError)
    setnewptr!(pyTabError, C.POINTERS.PyExc_TabError)
    setnewptr!(pyReferenceError, C.POINTERS.PyExc_ReferenceError)
    setnewptr!(pySystemError, C.POINTERS.PyExc_SystemError)
    setnewptr!(pySystemExit, C.POINTERS.PyExc_SystemExit)
    setnewptr!(pyTypeError, C.POINTERS.PyExc_TypeError)
    setnewptr!(pyUnboundLocalError, C.POINTERS.PyExc_UnboundLocalError)
    setnewptr!(pyUnicodeError, C.POINTERS.PyExc_UnicodeError)
    setnewptr!(pyUnicodeEncodeError, C.POINTERS.PyExc_UnicodeEncodeError)
    setnewptr!(pyUnicodeDecodeError, C.POINTERS.PyExc_UnicodeDecodeError)
    setnewptr!(pyUnicodeTranslateError, C.POINTERS.PyExc_UnicodeTranslateError)
    setnewptr!(pyValueError, C.POINTERS.PyExc_ValueError)
    setnewptr!(pyZeroDivisionError, C.POINTERS.PyExc_ZeroDivisionError)
    setnewptr!(pyBlockingIOError, C.POINTERS.PyExc_BlockingIOError)
    setnewptr!(pyBrokenPipeError, C.POINTERS.PyExc_BrokenPipeError)
    setnewptr!(pyChildProcessError, C.POINTERS.PyExc_ChildProcessError)
    setnewptr!(pyConnectionError, C.POINTERS.PyExc_ConnectionError)
    setnewptr!(pyConnectionAbortedError, C.POINTERS.PyExc_ConnectionAbortedError)
    setnewptr!(pyConnectionRefusedError, C.POINTERS.PyExc_ConnectionRefusedError)
    setnewptr!(pyFileExistsError, C.POINTERS.PyExc_FileExistsError)
    setnewptr!(pyFileNotFoundError, C.POINTERS.PyExc_FileNotFoundError)
    setnewptr!(pyInterruptedError, C.POINTERS.PyExc_InterruptedError)
    setnewptr!(pyIsADirectoryError, C.POINTERS.PyExc_IsADirectoryError)
    setnewptr!(pyNotADirectoryError, C.POINTERS.PyExc_NotADirectoryError)
    setnewptr!(pyPermissionError, C.POINTERS.PyExc_PermissionError)
    setnewptr!(pyProcessLookupError, C.POINTERS.PyExc_ProcessLookupError)
    setnewptr!(pyTimeoutError, C.POINTERS.PyExc_TimeoutError)
    setnewptr!(pyEnvironmentError, C.POINTERS.PyExc_EnvironmentError)
    setnewptr!(pyIOError, C.POINTERS.PyExc_IOError)
    setnewptr!(pyWarning, C.POINTERS.PyExc_Warning)
    setnewptr!(pyUserWarning, C.POINTERS.PyExc_UserWarning)
    setnewptr!(pyDeprecationWarning, C.POINTERS.PyExc_DeprecationWarning)
    setnewptr!(pyPendingDeprecationWarning, C.POINTERS.PyExc_PendingDeprecationWarning)
    setnewptr!(pySyntaxWarning, C.POINTERS.PyExc_SyntaxWarning)
    setnewptr!(pyRuntimeWarning, C.POINTERS.PyExc_RuntimeWarning)
    setnewptr!(pyFutureWarning, C.POINTERS.PyExc_FutureWarning)
    setnewptr!(pyImportWarning, C.POINTERS.PyExc_ImportWarning)
    setnewptr!(pyUnicodeWarning, C.POINTERS.PyExc_UnicodeWarning)
    setnewptr!(pyBytesWarning, C.POINTERS.PyExc_BytesWarning)
    setnewptr!(pyResourceWarning, C.POINTERS.PyExc_ResourceWarning)
    # BUILTINS NOT AVAILABLE FROM C
    b = pyimport("builtins")
    t = b.range; setnewptr!(pyrangetype, getptr(t)); pystolen!(t);
end
