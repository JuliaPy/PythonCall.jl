"""
    mutable struct Context

A handle to a loaded instance of libpython, its interpreter, function pointers, etc.
"""
@kwdef mutable struct Context
    is_embedded :: Bool = false
    is_initialized :: Bool = false
    is_preinitialized :: Bool = false
    lib_ptr :: Ptr{Cvoid} = C_NULL
    exe_path :: Union{String, Missing} = missing
    lib_path :: Union{String, Missing} = missing
    dlopen_flags :: UInt32 = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL
    pyprogname :: Union{String, Missing} = missing
    pyprogname_w :: Any = missing
    pyhome :: Union{String, Missing} = missing
    pyhome_w :: Any = missing
    which :: Symbol = :unknown # :CondaPkg, :PyCall, :embedded or :unknown
    version :: Union{VersionNumber, Missing} = missing
    matches_pycall :: Union{Bool, Missing} = missing
end

const CTX = Context()

function _atpyexit()
    if CTX.is_initialized && !CTX.is_preinitialized
        @warn "Python exited unexpectedly"
    end
    CTX.is_initialized = false
    return
end

function init_context()

    CTX.is_embedded = haskey(ENV, "JULIA_PYTHONCALL_LIBPTR")

    if CTX.is_embedded
        # In this case, getting a handle to libpython is easy
        CTX.lib_ptr = Ptr{Cvoid}(parse(UInt, ENV["JULIA_PYTHONCALL_LIBPTR"]))
        init_pointers()
        # Check Python is initialized
        Py_IsInitialized() == 0 && error("Python is not already initialized.")
        CTX.is_initialized = true
        CTX.which = :embedded
        exe_path = get(ENV, "JULIA_PYTHONCALL_EXE", "")
        if exe_path != ""
            CTX.exe_path = exe_path
        end
    else
        # Find Python executable
        exe_path = get(ENV, "JULIA_PYTHONCALL_EXE", "")
        if exe_path == "" || exe_path == "@CondaPkg"
            # By default, we use Python installed by CondaPkg.
            exe_path = Sys.iswindows() ? joinpath(CondaPkg.envdir(), "python.exe") : joinpath(CondaPkg.envdir(), "bin", "python")
            # It's not sufficient to only activate the env while Python is initialising,
            # it must also be active when loading extension modules (e.g. numpy). So we
            # activate the environment globally.
            # TODO: is this really necessary?
            CondaPkg.activate!(ENV)
            CTX.which = :CondaPkg
        elseif exe_path == "@PyCall"
            # PyCall compatibility mode
            PyCall = Base.require(PYCALL_PKGID)
            exe_path = PyCall.python::String
            CTX.lib_path = PyCall.libpython::String
            CTX.which = :PyCall
        elseif startswith(exe_path, "@")
            error("invalid JULIA_PYTHONCALL_EXE=$exe_path")
        else
            # Otherwise we use the Python specified
            CTX.which = :unknown
        end

        # Ensure Python is runnable
        try
            run(pipeline(`$exe_path --version`, stdout=devnull, stderr=devnull))
        catch
            error("Python executable $(repr(exe_path)) is not executable.")
        end
        CTX.exe_path = exe_path

        # For calling Python with UTF-8 IO
        function python_cmd(args)
            env = copy(ENV)
            env["PYTHONIOENCODING"] = "UTF-8"
            setenv(`$(CTX.exe_path) $args`, env)
        end

        # Find Python library
        lib_path = something(
            CTX.lib_path===missing ? nothing : CTX.lib_path,
            get(ENV, "JULIA_PYTHONCALL_LIB", nothing),
            Some(nothing)
        )
        if lib_path !== nothing
            lib_ptr = dlopen_e(lib_path, CTX.dlopen_flags)
            if lib_ptr == C_NULL
                error("Python library $(repr(lib_path)) could not be opened.")
            else
                CTX.lib_path = lib_path
                CTX.lib_ptr = lib_ptr
            end
        else
            for lib_path in readlines(python_cmd([joinpath(@__DIR__, "find_libpython.py"), "--list-all"]))
                lib_ptr = dlopen_e(lib_path, CTX.dlopen_flags)
                if lib_ptr == C_NULL
                    @warn "Python library $(repr(lib_path)) could not be opened."
                else
                    CTX.lib_path = lib_path
                    CTX.lib_ptr = lib_ptr
                    break
                end
            end
            CTX.lib_path === nothing && error("""
                Could not find Python library for Python executable $(repr(CTX.exe_path)).

                If you know where the library is, set environment variable 'JULIA_PYTHONCALL_LIB' to its path.
                """)
        end
        init_pointers()

        # Initialize
        with_gil() do
            CTX.is_preinitialized = Py_IsInitialized() != 0
            if CTX.is_preinitialized
                # Already initialized (maybe you're using PyCall as well)
                @assert CTX.which in (:embedded, :PyCall)
            else
                @assert CTX.which in (:unknown, :CondaPkg)
                # Find ProgramName and PythonHome
                script = if Sys.iswindows()
                    """
                    import sys
                    print(sys.executable)
                    if hasattr(sys, "base_exec_prefix"):
                        sys.stdout.write(sys.base_exec_prefix)
                    else:
                        sys.stdout.write(sys.exec_prefix)
                    """
                else
                    """
                    import sys
                    print(sys.executable)
                    if hasattr(sys, "base_exec_prefix"):
                        sys.stdout.write(sys.base_prefix)
                        sys.stdout.write(":")
                        sys.stdout.write(sys.base_exec_prefix)
                    else:
                        sys.stdout.write(sys.prefix)
                        sys.stdout.write(":")
                        sys.stdout.write(sys.exec_prefix)
                    """
                end
                CTX.pyprogname, CTX.pyhome = readlines(python_cmd(["-c", script]))

                # Set PythonHome
                CTX.pyhome_w = Base.cconvert(Cwstring, CTX.pyhome)
                Py_SetPythonHome(pointer(CTX.pyhome_w))

                # Set ProgramName
                CTX.pyprogname_w = Base.cconvert(Cwstring, CTX.pyprogname)
                Py_SetProgramName(pointer(CTX.pyprogname_w))

                # Start the interpreter and register exit hooks
                Py_InitializeEx(0)
                atexit() do
                    CTX.is_initialized = false
                    if CTX.version === missing || CTX.version < v"3.6"
                        Py_Finalize()
                    else
                        if Py_FinalizeEx() == -1
                            @warn "Py_FinalizeEx() error"
                        end
                    end
                end
            end
            CTX.is_initialized = true
            if Py_AtExit(@cfunction(_atpyexit, Cvoid, ())) == -1
                @warn "Py_AtExit() error"
            end
            if CTX.which != :embedded
                atexit() do
                    dlclose(CTX.lib_ptr)
                end
            end
        end
    end

    # Compare libpath with PyCall
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)

#     C.PyObject_TryConvert_AddRule("builtins.object", PyObject, CTryConvertRule_wrapref, -100)
#     C.PyObject_TryConvert_AddRule("builtins.object", PyRef, CTryConvertRule_wrapref, -200)
#     C.PyObject_TryConvert_AddRule("collections.abc.Sequence", PyList, CTryConvertRule_wrapref, 100)
#     C.PyObject_TryConvert_AddRule("collections.abc.Set", PySet, CTryConvertRule_wrapref, 100)
#     C.PyObject_TryConvert_AddRule("collections.abc.Mapping", PyDict, CTryConvertRule_wrapref, 100)
#     C.PyObject_TryConvert_AddRule("_io._IOBase", PyIO, CTryConvertRule_trywrapref, 100)
#     C.PyObject_TryConvert_AddRule("io.IOBase", PyIO, CTryConvertRule_trywrapref, 100)
#     C.PyObject_TryConvert_AddRule("<buffer>", PyArray, CTryConvertRule_trywrapref, 200)
#     C.PyObject_TryConvert_AddRule("<buffer>", Array, CTryConvertRule_PyArray_tryconvert, 0)
#     C.PyObject_TryConvert_AddRule("<buffer>", PyBuffer, CTryConvertRule_wrapref, -200)
#     C.PyObject_TryConvert_AddRule("<arrayinterface>", PyArray, CTryConvertRule_trywrapref, 200)
#     C.PyObject_TryConvert_AddRule("<arrayinterface>", Array, CTryConvertRule_PyArray_tryconvert, 0)
#     C.PyObject_TryConvert_AddRule("<arraystruct>", PyArray, CTryConvertRule_trywrapref, 200)
#     C.PyObject_TryConvert_AddRule("<arraystruct>", Array, CTryConvertRule_PyArray_tryconvert, 0)
#     C.PyObject_TryConvert_AddRule("<array>", PyArray, CTryConvertRule_trywrapref, 0)
#     C.PyObject_TryConvert_AddRule("<array>", Array, CTryConvertRule_PyArray_tryconvert, 0)

    with_gil() do

        # Get the python version
        verstr = Base.unsafe_string(Py_GetVersion())
        vermatch = match(r"^[0-9.]+", verstr)
        if vermatch === nothing
            error("Cannot parse version from version string: $(repr(verstr))")
        end
        CTX.version = VersionNumber(vermatch.match)
        v"3" ≤ CTX.version < v"4" || error(
            "Only Python 3 is supported, this is Python $(CTX.version) at $(CTX.exe_path===missing ? "unknown location" : CTX.exe_path).",
        )

    end

    @debug "Initialized PythonCall.jl" CTX.is_embedded CTX.is_initialized CTX.exe_path CTX.lib_path CTX.lib_ptr CTX.pyprogname CTX.pyhome CTX.version

    return
end

function Base.show(io::IO, ::MIME"text/plain", ctx::Context)
    show(io, typeof(io))
    print(io, ":")
    for k in fieldnames(Context)
        println(io)
        print(io, "  ", k, " = ")
        show(io, getfield(ctx, k))
    end
end

const PYTHONCALL_UUID = Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d")
const PYTHONCALL_PKGID = Base.PkgId(PYTHONCALL_UUID, "PythonCall")

const PYCALL_UUID = Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0")
const PYCALL_PKGID = Base.PkgId(PYCALL_UUID, "PyCall")

function init_pycall(PyCall::Module)
    # see if PyCall and PythonCall are using the same interpreter by checking if a couple of memory addresses are the same
    ptr1 = Py_GetVersion()
    ptr2 = @eval PyCall ccall(@pysym(:Py_GetVersion), Ptr{Cchar}, ())
    CTX.matches_pycall = ptr1 == ptr2
    if CTX.which == :PyCall
        @assert CTX.matches_pycall
    end
end
