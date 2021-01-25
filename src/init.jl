function __init__()
    # Check if libpython is already loaded (i.e. if the Julia interpreter was started from a Python process)
    CONFIG.isembedded = haskey(ENV, "PYTHONJL_LIBPTR")

    if CONFIG.isembedded
        # In this case, getting a handle to libpython is easy
        CONFIG.libptr = Ptr{Cvoid}(parse(UInt, ENV["PYTHONJL_LIBPTR"]))
        # Check Python is initialized
        C.Py_IsInitialized() == 0 && error("Python is not already initialized.")
    else
        # Find Python executable
        exepath = something(
            CONFIG.exepath,
            get(ENV, "PYTHONJL_EXE", nothing),
            Sys.which("python3"),
            Sys.which("python"),
            Some(nothing),
        )
        if exepath === nothing
            error(
                """
              Could not find Python executable.

              Ensure 'python3' or 'python' is in your PATH or set environment variable 'PYTHONJL_EXE'
              to the path to the Python executable.
              """,
            )
        end
        if exepath == "CONDA" || startswith(exepath, "CONDA:")
            CONFIG.isconda = true
            CONFIG.condaenv = exepath == "CONDA" ? Conda.ROOTENV : exepath[7:end]
            Conda._install_conda(CONFIG.condaenv)
            exepath = joinpath(
                Conda.python_dir(CONFIG.condaenv),
                Sys.iswindows() ? "python.exe" : "python",
            )
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

        # For calling Python with UTF-8 IO
        function python_cmd(args)
            env = copy(ENV)
            env["PYTHONIOENCODING"] = "UTF-8"
            setenv(`$(CONFIG.exepath) $args`, env)
        end

        # Find Python library
        libpath =
            something(CONFIG.libpath, get(ENV, "PYTHONJL_LIB", nothing), Some(nothing))
        if libpath !== nothing
            libptr = dlopen_e(path, CONFIG.dlopenflags)
            if libptr == C_NULL
                error("Python library $(repr(libpath)) could not be opened.")
            else
                CONFIG.libpath = libpath
                CONFIG.libptr = libptr
            end
        else
            for libpath in readlines(
                python_cmd([joinpath(@__DIR__, "find_libpython.py"), "--list-all"]),
            )
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

        # Check we are not already initialized
        C.Py_IsInitialized() == 0 || error("Python is already initialized.")

        # Initialize
        with_gil() do
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
            check(
                C.Py_AtExit(
                    @cfunction(() -> (CONFIG.isinitialized = false; nothing), Cvoid, ())
                ),
            )
            atexit() do
                CONFIG.isinitialized = false
                checkm1(C.Py_FinalizeEx())
            end
        end
    end

    C.PyObject_TryConvert_AddRules(
        "builtins.object",
        [(PyObject, CTryConvertRule_wrapref, -100), (PyRef, CTryConvertRule_wrapref, -200)],
    )
    C.PyObject_TryConvert_AddRules(
        "collections.abc.Sequence",
        [(PyList, CTryConvertRule_wrapref, 100)],
    )
    C.PyObject_TryConvert_AddRules(
        "collections.abc.Set",
        [(PySet, CTryConvertRule_wrapref, 100)],
    )
    C.PyObject_TryConvert_AddRules(
        "collections.abc.Mapping",
        [(PyDict, CTryConvertRule_wrapref, 100)],
    )
    C.PyObject_TryConvert_AddRules("_io._IOBase", [(PyIO, CTryConvertRule_trywrapref, 100)])
    C.PyObject_TryConvert_AddRules("io.IOBase", [(PyIO, CTryConvertRule_trywrapref, 100)])
    C.PyObject_TryConvert_AddRules(
        "<buffer>",
        [
            (PyArray, CTryConvertRule_trywrapref, 200),
            (PyBuffer, CTryConvertRule_wrapref, -200),
        ],
    )
    C.PyObject_TryConvert_AddRules(
        "<arrayinterface>",
        [(PyArray, CTryConvertRule_trywrapref, 200)],
    )
    C.PyObject_TryConvert_AddRules(
        "<arraystruct>",
        [(PyArray, CTryConvertRule_trywrapref, 200)],
    )
    C.PyObject_TryConvert_AddRules("<array>", [(PyArray, CTryConvertRule_trywrapref, 0)])

    with_gil() do

        @pyg `import sys, os`

        if !CONFIG.isembedded
            @py ```
            # Some modules expect sys.argv to be set
            sys.argv = [""]
            sys.argv.extend($ARGS)

            # Some modules test for interactivity by checking if sys.ps1 exists
            if $(isinteractive()) and not hasattr(sys, "ps1"):
                sys.ps1 = ">>> "
            ```
        end

        # Is this the same Python as in Conda?
        if !CONFIG.isconda &&
           haskey(ENV, "CONDA_PREFIX") &&
           isdir(ENV["CONDA_PREFIX"]) &&
           haskey(ENV, "CONDA_PYTHON_EXE") &&
           isfile(ENV["CONDA_PYTHON_EXE"]) &&
           realpath(ENV["CONDA_PYTHON_EXE"]) == realpath(
               CONFIG.exepath === nothing ? @pyv(`sys.executable`::String) : CONFIG.exepath,
           )

            CONFIG.isconda = true
            CONFIG.condaenv = ENV["CONDA_PREFIX"]
            CONFIG.exepath === nothing && (CONFIG.exepath = @pyv(`sys.executable`::String))
        end

        # Get the python version
        CONFIG.version =
            let (a, b, c, d, e) = @pyv(`sys.version_info`::Tuple{Int,Int,Int,String,Int})
                VersionNumber(a, b, c, (d,), (e,))
            end
        v"3" ≤ CONFIG.version < v"4" || error(
            "Only Python 3 is supported, this is Python $(CONFIG.version) at $(CONFIG.exepath===nothing ? "unknown location" : CONFIG.exepath).",
        )

        # EXPERIMENTAL: hooks to perform actions when certain modules are loaded
        if !CONFIG.isembedded
            @py ```
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
            qtfix_hook = $(() -> (CONFIG.qtfix && fix_qt_plugin_path(); nothing))
            JULIA_COMPAT_HOOKS.add_hook("PyQt4", qtfix_hook)
            JULIA_COMPAT_HOOKS.add_hook("PyQt5", qtfix_hook)
            JULIA_COMPAT_HOOKS.add_hook("PySide", qtfix_hook)
            JULIA_COMPAT_HOOKS.add_hook("PySide2", qtfix_hook)
            ```

            @require IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a" begin
                IJulia.push_postexecute_hook() do
                    CONFIG.pyplotautoshow && pyplotshow()
                end
            end
        end

        # EXPERIMENTAL: IPython integration
        if CONFIG.isembedded && CONFIG.ipythonintegration
            if !CONFIG.isipython
                @py ```
                try:
                    ok = "IPython" in sys.modules and sys.modules["IPython"].get_ipython() is not None
                except:
                    ok = False
                $(CONFIG.isipython::Bool) = ok
                ```
            end
            if CONFIG.isipython
                # Set `Base.stdout` to `sys.stdout` and ensure it is flushed after each execution
                @eval Base stdout = $(@pyv `sys.stdout`::PyIO)
                pushdisplay(TextDisplay(Base.stdout))
                pushdisplay(IPythonDisplay())
                @py ```
                mkcb = lambda cb: lambda: cb()
                sys.modules["IPython"].get_ipython().events.register("post_execute", mkcb($(() -> flush(Base.stdout))))
                ```
            end
        end
    end
end
