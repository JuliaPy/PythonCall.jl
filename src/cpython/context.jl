"""
    mutable struct Context

A handle to a loaded instance of libpython, its interpreter, function pointers, etc.
"""
mutable struct Context
    is_embedded :: Bool
    is_initialized :: Bool
    is_preinitialized :: Bool
    lib_ptr :: Ptr{Cvoid}
    exe_path :: Union{String, Missing}
    lib_path :: Union{String, Missing}
    dlopen_flags :: UInt32
    pyprogname :: Union{String, Missing}
    pyprogname_w :: Any
    pyhome :: Union{String, Missing}
    pyhome_w :: Any
    is_conda :: Bool
    conda_env :: Union{String, Missing}
    pointers :: CAPIPointers
    version :: Union{VersionNumber, Missing}
end

function Context(;
    is_embedded :: Union{Bool, Missing} = missing,
    lib_ptr :: Ptr{Cvoid} = C_NULL,
    exe_path :: Union{AbstractString, Missing} = missing,
    lib_path :: Union{AbstractString, Missing} = missing,
    dlopen_flags :: Integer = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL,
    pyprogname :: Union{AbstractString, Missing} = missing,
    pyhome :: Union{AbstractString, Missing} = missing,
    is_conda :: Union{Bool, Missing} = missing,
    conda_env :: Union{AbstractString, Missing} = missing,
)
    if is_embedded === missing
        is_embedded = haskey(ENV, "JULIA_PYTHONCALL_LIBPTR")
    end

    py = Context(is_embedded, false, false, lib_ptr, exe_path, lib_path, dlopen_flags, pyprogname, nothing, pyhome, nothing, false, conda_env, CAPIPointers(), missing)

    if py.is_embedded
        # In this case, getting a handle to libpython is easy
        if py.lib_ptr == C_NULL
            py.lib_ptr = Ptr{Cvoid}(parse(UInt, ENV["JULIA_PYTHONCALL_LIBPTR"]))
        end
        init!(py.pointers, py.lib_ptr)
        # Check Python is initialized
        py.Py_IsInitialized() == 0 && error("Python is not already initialized.")
        py.is_initialized = py.is_preinitialized = true
    elseif get(ENV, "JULIA_PYTHONCALL_EXE", "") == "PYCALL"
        error("not implemented: PyCall compatability mode")
#         # Import PyCall and use its choices for libpython
#         PyCall = get(Base.loaded_modules, PYCALL_PKGID, nothing)
#         if PyCall === nothing
#             PyCall = Base.require(PYCALL_PKGID)
#         end
#         CONFIG.exepath = PyCall.python
#         CONFIG.libpath = PyCall.libpython
#         CONFIG.libptr = dlopen_e(CONFIG.libpath, CONFIG.dlopenflags)
#         if CONFIG.libptr == C_NULL
#             error("Python library $(repr(CONFIG.libpath)) (from PyCall) could not be opened.")
#         end
#         CONFIG.pyprogname = PyCall.pyprogramname
#         CONFIG.pyhome = PyCall.PYTHONHOME
#         C.init_pointers()
#         # Check Python is initialized
#         C.Py_IsInitialized() == 0 && error("Python is not already initialized.")
#         CONFIG.isinitialized = CONFIG.preinitialized = true
    else
        # Find Python executable
        exe_path = something(
            py.exe_path===missing ? nothing : py.exe_path,
            get(ENV, "JULIA_PYTHONCALL_EXE", nothing),
            Sys.which("python3"),
            Sys.which("python"),
            get(ENV, "JULIA_PKGEVAL", "") == "true" ? "CONDA" : nothing,
            Some(nothing),
        )
        if exe_path === nothing
            error(
                """
              Could not find Python executable.

              Ensure 'python3' or 'python' is in your PATH or set environment variable 'JULIA_PYTHONCALL_EXE'
              to the path to the Python executable.
              """,
            )
        end
        if py.is_conda !== false && (exe_path == "CONDA" || startswith(exe_path, "CONDA:"))
            py.is_conda = true
            py.conda_env = exepath == "CONDA" ? Conda.ROOTENV : exe_path[7:end]
            Conda._install_conda(py.conda_env)
            exe_path = joinpath(
                Conda.python_dir(py.condaenv),
                Sys.iswindows() ? "python.exe" : "python",
            )
        end
        if isfile(exe_path)
            py.exe_path = exe_path
        else
            error("""
                Python executable $(repr(exe_path)) does not exist.

                Ensure either:
                - python3 or python is in your PATH
                - JULIA_PYTHONCALL_EXE is "CONDA", "CONDA:<env>" or "PYCALL"
                - JULIA_PYTHONCALL_EXE is the path to the Python executable
                """)
        end

        # For calling Python with UTF-8 IO
        function python_cmd(args)
            env = copy(ENV)
            env["PYTHONIOENCODING"] = "UTF-8"
            setenv(`$(py.exe_path) $args`, env)
        end

        # Find Python library
        lib_path = something(
            py.lib_path===missing ? nothing : py.lib_path,
            get(ENV, "JULIA_PYTHONCALL_LIB", nothing),
            Some(nothing)
        )
        if lib_path !== nothing
            lib_ptr = dlopen_e(lib_path, py.dlopen_flags)
            if lib_ptr == C_NULL
                error("Python library $(repr(lib_path)) could not be opened.")
            else
                py.lib_path = lib_path
                py.lib_ptr = lib_ptr
            end
        else
            for lib_path in readlines(python_cmd([joinpath(@__DIR__, "find_libpython.py"), "--list-all"]))
                lib_ptr = dlopen_e(lib_path, py.dlopen_flags)
                if lib_ptr == C_NULL
                    @warn "Python library $(repr(lib_path)) could not be opened."
                else
                    py.lib_path = lib_path
                    py.lib_ptr = lib_ptr
                    break
                end
            end
            py.lib_path === nothing && error("""
                Could not find Python library for Python executable $(repr(py.exe_path)).

                If you know where the library is, set environment variable 'JULIA_PYTHONCALL_LIB' to its path.
                """)
        end
        init!(py.pointers, py.lib_ptr)

        # # Compare libpath with PyCall
        # PyCall = get(Base.loaded_modules, PYCALL_PKGID, nothing)
        # if PyCall === nothing
        #     @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" check_libpath(PyCall)
        # else
        #     check_libpath(PyCall)
        # end

        # Initialize
        with_gil(py) do
            if py.Py_IsInitialized() != 0
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
                py.pyprogname, py.pyhome = readlines(python_cmd(["-c", script]))

                # Set PythonHome
                py.pyhome_w = Base.cconvert(Cwstring, py.pyhome)
                py.Py_SetPythonHome(pointer(py.pyhome_w))

                # Set ProgramName
                py.pyprogname_w = Base.cconvert(Cwstring, py.pyprogname)
                py.Py_SetProgramName(pointer(py.pyprogname_w))

                # Start the interpreter and register exit hooks
                py.Py_InitializeEx(0)
                # atexit() do
                #     py.is_initialized = false
                #     py.version < v"3.6" ? C.Py_Finalize() : checkm1(C.Py_FinalizeEx())
                # end
            end
            py.is_initialized = true
            # check(
            #     C.Py_AtExit(
            #         @cfunction(() -> (py.is_initialized = false; nothing), Cvoid, ())
            #     ),
            # )
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

    with_gil(py) do

#         @pyg `import sys, os`

#         pywordsize = (@pyv `sys.maxsize > 2**32`::Bool) ? 64 : 32
#         pywordsize == Sys.WORD_SIZE || error("Julia is $(Sys.WORD_SIZE)-bit but Python is $(pywordsize)-bit (at $(CONFIG.exepath ? "unknown location" : CONFIG.exepath))")

#         if !CONFIG.isembedded
#             @py ```
#             # Some modules expect sys.argv to be set
#             sys.argv = [""]
#             sys.argv.extend($ARGS)

#             # Some modules test for interactivity by checking if sys.ps1 exists
#             if $(isinteractive()) and not hasattr(sys, "ps1"):
#                 sys.ps1 = ">>> "
#             ```
#         end

        # # Is this the same Python as in Conda?
        # if py.is_conda &&
        #    haskey(ENV, "CONDA_PREFIX") &&
        #    isdir(ENV["CONDA_PREFIX"]) &&
        #    haskey(ENV, "CONDA_PYTHON_EXE") &&
        #    isfile(ENV["CONDA_PYTHON_EXE"]) &&
        #    realpath(ENV["CONDA_PYTHON_EXE"]) == realpath(
        #        py.exe_path === nothing ? @pyv(`sys.executable`::String) : py.exe_path,
        #    )

        #     py.isconda = true
        #     py.condaenv = ENV["CONDA_PREFIX"]
        #     py.exepath === nothing && (py.exepath = @pyv(`sys.executable`::String))
        # end

        # Get the python version
        py.version = VersionNumber(split(Base.unsafe_string(py.Py_GetVersion()), isspace)[1])
        v"3" ≤ py.version < v"4" || error(
            "Only Python 3 is supported, this is Python $(py.version) at $(py.exe_path===missing ? "unknown location" : py.exe_path).",
        )

#         # set up the 'juliacall' module
#         @py ```
#         import sys
#         if $(CONFIG.isembedded):
#             jl = sys.modules["juliacall"]
#         elif "juliacall" in sys.modules:
#             raise ImportError("'juliacall' module already exists")
#         else:
#             jl = sys.modules["juliacall"] = type(sys)("juliacall")
#             jl.CONFIG = dict()
#         jl.Main = $(pyjl(Main))
#         jl.Base = $(pyjl(Base))
#         jl.Core = $(pyjl(Core))
#         code = """
#         def newmodule(name):
#             "A new module with the given name."
#             return Base.Module(Base.Symbol(name))
#         class As:
#             "Interpret 'value' as type 'type' when converting to Julia."
#             __slots__ = ("value", "type")
#             def __init__(self, value, type):
#                 self.value = value
#                 self.type = type
#             def __repr__(self):
#                 return "juliacall.As({!r}, {!r})".format(self.value, self.type)
#         """
#         exec(code, jl.__dict__)
#         ```

#         # EXPERIMENTAL: hooks to perform actions when certain modules are loaded
#         if !CONFIG.isembedded
#             @py ```
#             import sys
#             class JuliaCompatHooks:
#                 def __init__(self):
#                     self.hooks = {}
#                 def find_module(self, name, path=None):
#                     hs = self.hooks.get(name)
#                     if hs is not None:
#                         for h in hs:
#                             h()
#                 def add_hook(self, name, h):
#                     if name not in self.hooks:
#                         self.hooks[name] = [h]
#                     else:
#                         self.hooks[name].append(h)
#                     if name in sys.modules:
#                         h()
#             JULIA_COMPAT_HOOKS = JuliaCompatHooks()
#             sys.meta_path.insert(0, JULIA_COMPAT_HOOKS)

#             # Before Qt is loaded, fix the path used to look up its plugins
#             qtfix_hook = $(() -> (CONFIG.qtfix && fix_qt_plugin_path(); nothing))
#             JULIA_COMPAT_HOOKS.add_hook("PyQt4", qtfix_hook)
#             JULIA_COMPAT_HOOKS.add_hook("PyQt5", qtfix_hook)
#             JULIA_COMPAT_HOOKS.add_hook("PySide", qtfix_hook)
#             JULIA_COMPAT_HOOKS.add_hook("PySide2", qtfix_hook)
#             ```

#             @require IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a" begin
#                 IJulia.push_postexecute_hook() do
#                     CONFIG.pyplotautoshow && pyplotshow()
#                 end
#             end
#         end

#         # EXPERIMENTAL: IPython integration
#         if CONFIG.isembedded && CONFIG.ipythonintegration
#             if !CONFIG.isipython
#                 @py ```
#                 try:
#                     ok = "IPython" in sys.modules and sys.modules["IPython"].get_ipython() is not None
#                 except:
#                     ok = False
#                 $(CONFIG.isipython::Bool) = ok
#                 ```
#             end
#             if CONFIG.isipython
#                 # Set `Base.stdout` to `sys.stdout` and ensure it is flushed after each execution
#                 @eval Base stdout = $(@pyv `sys.stdout`::PyIO)
#                 pushdisplay(TextDisplay(Base.stdout))
#                 pushdisplay(IPythonDisplay())
#                 @py ```
#                 mkcb = lambda cb: lambda: cb()
#                 sys.modules["IPython"].get_ipython().events.register("post_execute", mkcb($(() -> flush(Base.stdout))))
#                 ```
#             end
#         end
    end

    @debug "Initialized PythonCall.jl" py.is_embedded py.is_initialized py.exe_path py.lib_path py.lib_ptr py.pyprogname py.pyhome py.version, py.is_conda py.conda_env
    return py
end

function Base.show(io::IO, py::Context)
    show(io, typeof(py))
    print(io, "(exe_path=")
    show(io, py.exe_path)
    print(io, ", lib_path=")
    show(io, py.lib_path)
    print(io, ", ...)")
end

const PYCALL_UUID = Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0")
const PYCALL_PKGID = Base.PkgId(PYCALL_UUID, "PyCall")

check_libpath(PyCall, py) = begin
    if realpath(PyCall.libpython) == realpath(py.lib_path)
        # @info "libpython path agrees between PythonCall and PyCall" PythonCall.CONFIG.libpath PyCall.libpython
    else
        @warn "PythonCall and PyCall are using different versions of libpython. This will probably go badly." py.lib_path PyCall.libpython
    end
end

struct Func{name}
    ctx :: Context
end

Base.getproperty(py::Context, k::Symbol) = hasfield(Context, k) ? getfield(py, k) : Func{k}(py)

const CONTEXT_PROPERTYNAMES = (fieldnames(Context)..., CAPI_FUNCS..., :with_gil)

Base.propertynames(::Context, private::Bool=false) = CONTEXT_PROPERTYNAMES

for (name, (argtypes, rettype)) in CAPI_FUNC_SIGS
    args = [Symbol("x", i) for (i,_) in enumerate(argtypes)]
    functype = Func{name}
    @eval $name(py::Context, $(args...)) = ccall(py.pointers.$name, $rettype, ($(argtypes...),), $(args...))
    @eval (f::$functype)($(args...)) = $name(f.ctx, $(args...))
end
