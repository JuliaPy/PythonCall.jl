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
    version :: Union{VersionNumber, Missing} = missing
end

const CTX = Context()

function init_context()

    CTX.is_embedded = haskey(ENV, "JULIA_PYTHONCALL_LIBPTR")

    if CTX.is_embedded
        # In this case, getting a handle to libpython is easy
        CTX.lib_ptr = Ptr{Cvoid}(parse(UInt, ENV["JULIA_PYTHONCALL_LIBPTR"]))
        init_pointers()
        # Check Python is initialized
        Py_IsInitialized() == 0 && error("Python is not already initialized.")
        CTX.is_initialized = CTX.is_preinitialized = true
    else
        # Find Python executable
        # TODO: PyCall compatibility mode
        # TODO: when JULIA_PYTHONCALL_EXE is given, determine if we are in a conda environment
        exe_path = get(ENV, "JULIA_PYTHONCALL_EXE", "")
        if exe_path == ""
            # By default, we use a conda environment inside the first Julia environment in
            # the LOAD_PATH in which PythonCall is installed (in the manifest as an
            # indirect dependency).
            #
            # Note that while Julia environments are stacked, Python environments are not,
            # so it is possible that two Julia environments contain two different packages
            # depending on PythonCall which install into this one conda environment.
            #
            # Regarding the LOAD_PATH as getting "less specific" as we go through, this
            # choice of location is the "most specific" place which actually depends on
            # PythonCall.
            conda_env = nothing
            for env in Base.load_path()
                proj = Base.env_project_file(env)
                is_pythoncall = Base.project_file_name_uuid(proj, "").uuid == PYTHONCALL_UUID
                depends_on_pythoncall = Base.manifest_uuid_path(env, PYTHONCALL_PKGID) !== nothing
                if is_pythoncall || depends_on_pythoncall
                    envdir = proj isa String ? dirname(proj) : env
                    conda_env = Conda._env[] = joinpath(envdir, ".conda_env")
                    break
                end
            end
            conda_env isa String || error("could not find the environment containing PythonCall (this is a bug, please report it)")
            # ensure the environment exists
            if !isdir(conda_env)
                @info "Creating conda environment" conda_env
                Conda.create()
            end
            # activate
            Conda.activate()
            # ensure python exists
            exe_path = Conda.python_exe()
            if !isfile(exe_path)
                @info "Installing Python"
                Conda.add("python")
                isfile(exe_path) || error("installed python but still can't find it")
            end
        end

        # Ensure Python us runnable
        try
            run(pipeline(`$exe_path --version`, devnull))
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

        # Compare libpath with PyCall
        PyCall = get(Base.loaded_modules, PYCALL_PKGID, nothing)
        if PyCall === nothing
            @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" check_libpath(PyCall)
        else
            check_libpath(PyCall)
        end

        # Initialize
        with_gil() do
            if Py_IsInitialized() != 0
                # Already initialized (maybe you're using PyCall as well)
            else
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
            if Py_AtExit(@cfunction(() -> (CTX.is_initialized && @warn("Python exited unexpectedly"); CTX.is_initialized = false; nothing), Cvoid, ())) == -1
                @warn "Py_AtExit() error"
            end
        end
    end

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

    @debug "Initialized PythonCall.jl" CTX.is_embedded CTX.is_initialized CTX.exe_path CTX.lib_path CTX.lib_ptr CTX.pyprogname CTX.pyhome CTX.version Conda._env[]

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

check_libpath(PyCall) = begin
    if realpath(PyCall.libpython) == realpath(CTX.lib_path)
        # @info "libpython path agrees between PythonCall and PyCall" PythonCall.CONFIG.libpath PyCall.libpython
    else
        @warn "PythonCall and PyCall are using different versions of libpython. This will probably go badly." CTX.lib_path PyCall.libpython
    end
end
