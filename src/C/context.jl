"""
    mutable struct Context

A handle to a loaded instance of libpython, its interpreter, function pointers, etc.
"""
@kwdef mutable struct Context
    is_embedded::Bool = false
    is_initialized::Bool = false
    is_preinitialized::Bool = false
    lib_ptr::Ptr{Cvoid} = C_NULL
    exe_path::Union{String,Missing} = missing
    lib_path::Union{String,Missing} = missing
    dlopen_flags::UInt32 = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL
    pyprogname::Union{String,Missing} = missing
    pyprogname_w::Any = missing
    pyhome::Union{String,Missing} = missing
    pyhome_w::Any = missing
    which::Symbol = :unknown # :CondaPkg, :PyCall, :embedded or :unknown
    version::Union{VersionNumber,Missing} = missing
    matches_pycall::Union{Bool,Missing} = missing
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

    CTX.is_embedded = hasproperty(Base.Main, :__PythonCall_libptr)

    if CTX.is_embedded
        # In this case, getting a handle to libpython is easy
        CTX.lib_ptr = Base.Main.__PythonCall_libptr::Ptr{Cvoid}
        init_pointers()
        # Check Python is initialized
        Py_IsInitialized() == 0 && error("Python is not already initialized.")
        CTX.is_initialized = true
        CTX.which = :embedded
        exe_path = get(ENV, "JULIA_PYTHONCALL_EXE", "")
        if exe_path != ""
            CTX.exe_path = exe_path
            # this ensures PyCall uses the same Python interpreter
            get!(ENV, "PYTHON", exe_path)
        end
    else
        # Find Python executable
        exe_path = get(ENV, "JULIA_PYTHONCALL_EXE", "")
        if exe_path == "" || exe_path == "@CondaPkg"
            if CondaPkg.backend() == :Null
                exe_path = Sys.which("python")
                if exe_path === nothing
                    error("CondaPkg is using the Null backend but Python is not installed")
                end
                exe_path::String
            else
                # By default, we use Python installed by CondaPkg.
                exe_path =
                    Sys.iswindows() ? joinpath(CondaPkg.envdir(), "python.exe") :
                    joinpath(CondaPkg.envdir(), "bin", "python")
                # It's not sufficient to only activate the env while Python is initialising,
                # it must also be active when loading extension modules (e.g. numpy). So we
                # activate the environment globally.
                # TODO: is this really necessary?
                CondaPkg.activate!(ENV)
            end
            CTX.which = :CondaPkg
        elseif exe_path == "@PyCall"
            # PyCall compatibility mode
            PyCall = Base.require(PYCALL_PKGID)::Module
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
            run(pipeline(`$exe_path --version`, stdout = devnull, stderr = devnull))
        catch
            error("Python executable $(repr(exe_path)) is not executable.")
        end
        CTX.exe_path = exe_path

        # For calling Python with UTF-8 IO
        function python_cmd(args)
            env = copy(ENV)
            env["PYTHONIOENCODING"] = "UTF-8"
            delete!(env, "PYTHONHOME")
            setenv(`$(CTX.exe_path) $args`, env)
        end

        # Find and open Python library
        lib_path = something(
            CTX.lib_path === missing ? nothing : CTX.lib_path,
            get(ENV, "JULIA_PYTHONCALL_LIB", nothing),
            Some(nothing),
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
            for lib_path in readlines(
                python_cmd([joinpath(@__DIR__, "find_libpython.py"), "--list-all"]),
            )
                lib_ptr = dlopen_e(lib_path, CTX.dlopen_flags)
                if lib_ptr == C_NULL
                    @warn "Python library $(repr(lib_path)) could not be opened."
                else
                    CTX.lib_path = lib_path
                    CTX.lib_ptr = lib_ptr
                    break
                end
            end
            CTX.lib_path === missing && error("""
                Could not find Python library for Python executable $(repr(CTX.exe_path)).

                If you know where the library is, set environment variable 'JULIA_PYTHONCALL_LIB' to its path.
                """)
        end

        # Close the library when Julia exits
        atexit() do
            dlclose(CTX.lib_ptr)
        end

        # Get function pointers from the library
        init_pointers()

        # Compare libpath with PyCall
        @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" init_pycall(PyCall)

        # Initialize the interpreter
        CTX.is_preinitialized = Py_IsInitialized() != 0
        if CTX.is_preinitialized
            @assert CTX.which == :PyCall || CTX.matches_pycall isa Bool
        else
            @assert CTX.which != :PyCall
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
    end

    # HACK: If we are using CondaPkg, prevent child processes from using it by explicitly
    # setting the executable. Our tests sometimes fail on Windows because Aqua launches a
    # child process which causes CondaPkg to re-resolve (should work out why, I assume it's
    # a filesystem timestamp thing) which causes some stdlibs to disappear for a bit.
    #
    # A better solution may be to use some environment variable to "freeze" CondaPkg in
    # child processes.
    # 
    # Only done when CI=true since it's a hack.
    if (get(ENV, "CI", "false") == "true") && (CTX.which === :CondaPkg)
        ENV["JULIA_PYTHONCALL_EXE"] = CTX.exe_path::String
    end

    # Get the python version
    verstr = Base.unsafe_string(Py_GetVersion())
    vermatch = match(r"^[0-9.]+", verstr)
    if vermatch === nothing
        error("Cannot parse version from version string: $(repr(verstr))")
    end
    CTX.version = VersionNumber(vermatch.match)
    v"3.5" ≤ CTX.version < v"4" || error(
        "Only Python 3.5+ is supported, this is Python $(CTX.version) at $(CTX.exe_path===missing ? "unknown location" : CTX.exe_path).",
    )

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
