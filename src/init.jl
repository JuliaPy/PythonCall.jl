function python_cmd(args)
    env = copy(ENV)
    env["PYTHONIOENCODING"] = "UTF-8"
    setenv(`$(CONFIG.exepath) $args`, env)
end

function __init__()
    # Check if libpython is already loaded (i.e. if the Julia interpreter was started from a Python process)
    CONFIG.preloaded = try
        cglobal(:Py_Initialize)
        true
    catch
        false
    end

    # If not loaded, load it
    if !CONFIG.preloaded
        # Find Python executable
        exepath = something(CONFIG.exepath, get(ENV, "PYTHONJL_EXE", nothing), Sys.which("python3"), Sys.which("python"), Some(nothing))
        if exepath === nothing
            error("""
                Could not find Python executable.

                Ensure 'python3' or 'python' is in your PATH or set environment variable 'PYTHONJL_EXE'
                to the path to the Python executable.
                """)
        end
        if exepath == "CONDA" || startswith(exepath, "CONDA:")
            CONFIG.isconda = true
            CONFIG.condaenv = exepath == "CONDA" ? Conda.ROOTENV : exepath[7:end]
            Conda._install_conda(CONFIG.condaenv)
            exepath = joinpath(Conda.python_dir(CONFIG.condaenv), Sys.iswindows() ? "python.exe" : "python")
        end
        if isfile(exepath)
            CONFIG.exepath = exepath
        else
            error("""
                Python executable $(repr(exepath)) does not exist.

                Ensure either:
                - python3 or python is in your PATH
                - PYTHONJL_EXE is "CONDA" or "CONDA:<env>"
                - PYTHONJL_EXE is the path to the Python executable
                """)
        end

        # Find Python library
        libpath = something(CONFIG.libpath, get(ENV, "PYTHONJL_LIB", nothing), Some(nothing))
        if libpath !== nothing
            libptr = dlopen_e(path, CONFIG.dlopenflags)
            if libptr == C_NULL
                error("Python library $(repr(libpath)) could not be opened.")
            else
                CONFIG.libpath = libpath
                CONFIG.libptr = libptr
            end
        else
            for libpath in readlines(python_cmd([joinpath(@__DIR__, "find_libpython.py"), "--list-all"]))
                libptr = dlopen_e(libpath, CONFIG.dlopenflags)
                if libptr == C_NULL
                    @warn "Python library $(repr(libpath)) could not be opened."
                else
                    CONFIG.libpath = libpath
                    CONFIG.libptr = libptr
                    break
                end
            end
            CONFIG.libpath === nothing && error("""
                Could not find Python library for Python executable $(repr(CONFIG.exepath)).

                If you know where the library is, set environment variable 'PYTHONJL_LIB' to its path.
                """)
        end
    end

    # Check if libpython is already initialized
    CONFIG.preinitialized = C.Py_IsInitialized() != 0

    # If not, initialize it
    if !CONFIG.preinitialized

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
        CONFIG.pyprogname, CONFIG.pyhome = readlines(python_cmd(["-c", script]))

        # Set PythonHome
        CONFIG.pyhome_w = Base.cconvert(Cwstring, CONFIG.pyhome)
        C.Py_SetPythonHome(pointer(CONFIG.pyhome_w))

        # Set ProgramName
        CONFIG.pyprogname_w = Base.cconvert(Cwstring, CONFIG.pyprogname)
        C.Py_SetProgramName(pointer(CONFIG.pyprogname_w))

        # Start the interpreter and register exit hooks
        C.Py_InitializeEx(0)
        CONFIG.isinitialized = true
        check(C.Py_AtExit(@cfunction(()->(CONFIG.isinitialized = false; nothing), Cvoid, ())))
        atexit() do
            CONFIG.isinitialized = false
            check(C.Py_FinalizeEx())
        end

        # Some modules expect sys.argv to be set
        pysysmodule.argv = pylist([""; ARGS])

        # Some modules test for interactivity by checking if sys.ps1 exists
        if isinteractive() && !pyhasattr(pysysmodule, "ps1")
            pysysmodule.ps1 = ">>> "
        end
    end

    # Is this the same Python as in Conda?
    if !CONFIG.isconda && isdir(Conda.PREFIX) && realpath(pyconvert(String, pysysmodule.prefix)) == realpath(Conda.PREFIX)
        CONFIG.isconda = true
        CONFIG.condaenv = Conda.ROOTENV
    end

    # Get the python version
    CONFIG.version = let (a,b,c,d,e) = pyconvert(Tuple{Int,Int,Int,String,Int}, pysysmodule.version_info)
        VersionNumber(a, b, c, (d,), (e,))
    end
    v"2" < CONFIG.version < v"4" || error("Only Python 3 is supported, this is Python $(CONFIG.version.major).$(CONFIG.version.minor) at $(CONFIG.preloaded ? CONFIG.exepath : "unknown location").")

    # EXPERIMENTAL: hooks to perform actions when certain modules are loaded
    if !CONFIG.preinitialized
        py"""
        import sys
        class JuliaCompatHooks:
            def __init__(self):
                self.hooks = {}
            def find_module(self, name, path=None):
                hs = self.hooks.get(name)
                if hs is not None:
                    for h in hs:
                        h()
            def add_hook(self, name, h):
                if name not in self.hooks:
                    self.hooks[name] = [h]
                else:
                    self.hooks[name].append(h)
                if name in sys.modules:
                    h()
        JULIA_COMPAT_HOOKS = JuliaCompatHooks()
        sys.meta_path.insert(0, JULIA_COMPAT_HOOKS)

        # Before Qt is loaded, fix the path used to look up its plugins
        qtfix_hook = $(pyjlfunction(() -> if CONFIG.qtfix; fix_qt_plugin_path(); nothing; end))
        JULIA_COMPAT_HOOKS.add_hook("PyQt4", qtfix_hook)
        JULIA_COMPAT_HOOKS.add_hook("PyQt5", qtfix_hook)
        JULIA_COMPAT_HOOKS.add_hook("PySide", qtfix_hook)
        JULIA_COMPAT_HOOKS.add_hook("PySide2", qtfix_hook)
        """

        @require IJulia="7073ff75-c697-5162-941a-fcdaad2a7d2a" begin
            IJulia.push_postexecute_hook() do
                if CONFIG.pyplotautoshow && "matplotlib.pyplot" in pysysmodule.modules
                    pyplotshow()
                end
            end
        end
    end

    return
end
