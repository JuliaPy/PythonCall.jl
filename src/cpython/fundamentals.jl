### INITIALIZE

Py_Initialize() = ccall(POINTERS.Py_Initialize, Cvoid, ())
Py_InitializeEx(flag) = ccall(POINTERS.Py_InitializeEx, Cvoid, (Cint,), flag)
Py_Finalize() = ccall(POINTERS.Py_Finalize, Cvoid, ())
Py_FinalizeEx() = ccall(POINTERS.Py_FinalizeEx, Cint, ())
Py_AtExit(callback) = ccall(POINTERS.Py_AtExit, Cint, (Ptr{Cvoid},), callback)
Py_IsInitialized() = ccall(POINTERS.Py_IsInitialized, Cint, ())
Py_SetPythonHome(name) = ccall(POINTERS.Py_SetPythonHome, Cvoid, (Cwstring,), name)
Py_SetProgramName(name) = ccall(POINTERS.Py_SetProgramName, Cvoid, (Cwstring,), name)
Py_GetVersion() = ccall(POINTERS.Py_GetVersion, Cstring, ())

### REFCOUNT

_Py_IncRef(o) = ccall(POINTERS.Py_IncRef, Cvoid, (PyPtr,), o)
_Py_DecRef(o) = ccall(POINTERS.Py_DecRef, Cvoid, (PyPtr,), o)
const FAST_INCREF = true
const FAST_DECREF = false
if FAST_INCREF
    # This avoids calling the C-API Py_IncRef().
    # It just needs to increase the reference count.
    # Assumes Python is not built for debugging reference counts.
    # Speed up from 2.5ns to 1.3ns.
    Py_IncRef(o) = GC.@preserve o begin
        p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
        if p != C_NULL
            p.refcnt[] += 1
        end
        nothing
    end
else
    Py_IncRef(o) = _Py_IncRef(o)
end
if FAST_DECREF
    # This avoids calling the C-API Py_IncRef() unless the object is about to be deallocated.
    # It just needs to decrement the reference count.
    # Assumes Python is not built for debugging reference counts.
    # Speed up from 2.5ns to 1.8ns in non-deallocating case.
    Py_DecRef(o) = GC.@preserve o begin
        p = UnsafePtr(Base.unsafe_convert(PyPtr, o))
        if p != C_NULL
            c = p.refcnt[]
            if c > 1
                p.refcnt[] = c - 1
            else
                _Py_DecRef(o)
            end
        end
        nothing
    end
else
    Py_DecRef(o) = _Py_DecRef(o)
end
Py_RefCnt(o) = GC.@preserve o UnsafePtr(Base.unsafe_convert(PyPtr, o)).refcnt[]

Py_DecRef(f::Function, o::Ptr, dflt = PYERR()) =
    isnull(o) ? dflt : (r = f(o); Py_DecRef(o); r)

Py_Is(o1, o2) = Base.unsafe_convert(PyPtr, o1) == Base.unsafe_convert(PyPtr, o2)

### EVAL

PyEval_EvalCode(code, globals, locals) = ccall(POINTERS.PyEval_EvalCode, PyPtr, (PyPtr, PyPtr, PyPtr), code, globals, locals)
Py_CompileString(code, filename, mode) = ccall(POINTERS.Py_CompileString, PyPtr, (Cstring, Cstring, Cint), code, filename, mode)
PyEval_GetBuiltins() = ccall(POINTERS.PyEval_GetBuiltins, PyPtr, ())
PyRun_InteractiveOne(file, filename) = ccall(POINTERS.PyRun_InteractiveOne, Cint, (Ptr{Cvoid}, Cstring), file, filename) # (FILE* file, const char* filename)

### GIL & THREADS

PyEval_SaveThread() = ccall(POINTERS.PyEval_SaveThread, Ptr{Cvoid}, ())
PyEval_RestoreThread(ptr) = ccall(POINTERS.PyEval_RestoreThread, Cvoid, (Ptr{Cvoid},), ptr)
PyGILState_Ensure() = ccall(POINTERS.PyGILState_Ensure, PyGILState_STATE, ())
PyGILState_Release(state) = ccall(POINTERS.PyGILState_Release, Cvoid, (PyGILState_STATE,), state)

### IMPORT

PyImport_ImportModule(name) = ccall(POINTERS.PyImport_ImportModule, PyPtr, (Cstring,), name)
PyImport_Import(name) = ccall(POINTERS.PyImport_Import, PyPtr, (PyPtr,), name)
PyImport_GetModuleDict() = ccall(POINTERS.PyImport_GetModuleDict, PyPtr, ())
PyImport_GetModule(name) = begin
    ms = PyImport_GetModuleDict()
    ok = PyMapping_HasKeyString(ms, name)
    ism1(ok) && return PyNULL
    ok != 0 ? PyMapping_GetItemString(ms, name) : PyNULL
end

### MODULES

PyModule_GetDict(mod) = ccall(POINTERS.PyModule_GetDict, PyPtr, (PyPtr,), mod)

### ERRORS

PyErr_Occurred() = ccall(POINTERS.PyErr_Occurred, PyPtr, ())
PyErr_GivenExceptionMatches(exc, typ) = ccall(POINTERS.PyErr_GivenExceptionMatches, Cint, (PyPtr, PyPtr), exc, typ)
PyErr_Clear() = ccall(POINTERS.PyErr_Clear, Cvoid, ())
PyErr_SetNone(typ) = ccall(POINTERS.PyErr_SetNone, Cvoid, (PyPtr,), typ)
PyErr_SetString(typ, value) = ccall(POINTERS.PyErr_SetString, Cvoid, (PyPtr, Cstring), typ, value)
PyErr_SetObject(typ, value) = ccall(POINTERS.PyErr_SetObject, Cvoid, (PyPtr, PyPtr), typ, value)
PyErr_Fetch(t, v, b) = ccall(POINTERS.PyErr_Fetch, Cvoid, (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr}), t, v, b)
PyErr_NormalizeException(t, v, b) = ccall(POINTERS.PyErr_NormalizeException, Cvoid, (Ptr{PyPtr}, Ptr{PyPtr}, Ptr{PyPtr}), t, v, b)
PyErr_Restore(t, v, b) = ccall(POINTERS.PyErr_Restore, Cvoid, (PyPtr, PyPtr, PyPtr), t, v, b)

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

for name in CAPI_EXCEPTIONS
    @eval $name() = POINTERS.$name
end

### INPUT HOOK

function PyOS_RunInputHook()
    hook = pyloadglobal(Ptr{Cvoid}, :PyOS_InputHook)
    isnull(hook) || ccall(hook, Cint, ())
    nothing
end
