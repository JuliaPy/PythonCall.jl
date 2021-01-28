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

Py_DecRef(f::Function, o::Ptr, dflt = PYERR()) =
    isnull(o) ? dflt : (r = f(o); Py_DecRef(o); r)

Py_Is(o1, o2) = Base.unsafe_convert(PyPtr, o1) == Base.unsafe_convert(PyPtr, o2)

### EVAL

@cdef :PyEval_EvalCode PyPtr (PyPtr, PyPtr, PyPtr)
@cdef :Py_CompileString PyPtr (Cstring, Cstring, Cint)
@cdef :PyEval_GetBuiltins PyPtr ()
@cdef :PyRun_InteractiveOne Cint (Ptr{Cvoid}, Cstring) # (FILE* file, const char* filename)

### GIL & THREADS

@cdef :PyEval_SaveThread Ptr{Cvoid} ()
@cdef :PyEval_RestoreThread Cvoid (Ptr{Cvoid},)
@cdef :PyGILState_Ensure PyGILState_STATE ()
@cdef :PyGILState_Release Cvoid (PyGILState_STATE,)

### IMPORT

@cdef :PyImport_ImportModule PyPtr (Cstring,)
@cdef :PyImport_Import PyPtr (PyPtr,)
@cdef :PyImport_GetModuleDict PyPtr ()

PyImport_GetModule(name) = begin
    ms = PyImport_GetModuleDict()
    ok = PyMapping_HasKeyString(ms, name)
    ism1(ok) && return PyNULL
    ok != 0 ? PyMapping_GetItemString(ms, name) : PyNULL
end

### MODULES

@cdef :PyModule_GetDict PyPtr (PyPtr,)

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

PyErr_IsSet(t) =
    (o = PyErr_Occurred(); !isnull(o) && PyErr_GivenExceptionMatches(o, t) != 0)

function PyErr_FetchTuple(normalize::Bool = false)
    t = Ref{PyPtr}()
    v = Ref{PyPtr}()
    b = Ref{PyPtr}()
    PyErr_Fetch(t, v, b)
    normalize && PyErr_NormalizeException(t, v, b)
    (t[], v[], b[])
end

### EXCEPTIONS

for x in [
    :BaseException,
    :Exception,
    :StopIteration,
    :GeneratorExit,
    :ArithmeticError,
    :LookupError,
    :AssertionError,
    :AttributeError,
    :BufferError,
    :EOFError,
    :FloatingPointError,
    :OSError,
    :ImportError,
    :IndexError,
    :KeyError,
    :KeyboardInterrupt,
    :MemoryError,
    :NameError,
    :OverflowError,
    :RuntimeError,
    :RecursionError,
    :NotImplementedError,
    :SyntaxError,
    :IndentationError,
    :TabError,
    :ReferenceError,
    :SystemError,
    :SystemExit,
    :TypeError,
    :UnboundLocalError,
    :UnicodeError,
    :UnicodeEncodeError,
    :UnicodeDecodeError,
    :UnicodeTranslateError,
    :ValueError,
    :ZeroDivisionError,
    :BlockingIOError,
    :BrokenPipeError,
    :ChildProcessError,
    :ConnectionError,
    :ConnectionAbortedError,
    :ConnectionRefusedError,
    :FileExistsError,
    :FileNotFoundError,
    :InterruptedError,
    :IsADirectoryError,
    :NotADirectoryError,
    :PermissionError,
    :ProcessLookupError,
    :TimeoutError,
    :EnvironmentError,
    :IOError,
    :WindowsError,
    :Warning,
    :UserWarning,
    :DeprecationWarning,
    :PendingDeprecationWarning,
    :SyntaxWarning,
    :RuntimeWarning,
    :FutureWarning,
    :ImportWarning,
    :UnicodeWarning,
    :BytesWarning,
    :ResourceWarning,
]
    f = Symbol(:PyExc_, x)
    r = Symbol(f, :__ref)
    @eval const $r = Ref(PyNULL)
    @eval $f() = pyloadglobal($r, $(QuoteNode(f)))
end

### INPUT HOOK

function PyOS_RunInputHook()
    hook = unsafe_load(Ptr{Ptr{Cvoid}}(pyglobal(:PyOS_InputHook)))
    isnull(hook) || ccall(hook, Cint, ())
    nothing
end
