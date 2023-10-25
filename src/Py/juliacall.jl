const pyjuliacallmodule = pynew()
const pyJuliaError = pynew()
const CPyExc_JuliaError = Ref(C.PyNULL)

function init_juliacall()
    # ensure the 'juliacall' module exists
    # this means that Python code can do "import juliacall" and it will work regardless of
    # whether Python is embedded in Julia or vice-versa
    jl = pyjuliacallmodule
    sys = pysysmodule
    os = pyosmodule
    if C.CTX.is_embedded
        # in this case, Julia is being embedded into Python by juliacall, which already exists
        pycopy!(jl, sys.modules["juliacall"])
        @assert pystr_asstring(jl.__version__) == string(VERSION)
    elseif "juliacall" in sys.modules
        # otherwise, Python is being embedded into Julia by PythonCall, so should not exist
        error("'juliacall' module already exists")
    else
        # TODO: Is there a more robust way to import juliacall from a specific path?
        # prepend the directory containing juliacall to sys.path
        sys.path.insert(0, joinpath(ROOT_DIR, "pysrc"))
        # prevent juliacall from initialising itself
        os.environ["PYTHON_JULIACALL_INIT"] = "no"
        # import juliacall
        pycopy!(jl, pyimport("juliacall"))
        # check the version
        @assert realpath(pystr_asstring(jl.__path__[0])) == realpath(joinpath(ROOT_DIR, "pysrc", "juliacall"))
        @assert pystr_asstring(jl.__version__) == string(VERSION)
        @assert !pybool_asbool(jl.CONFIG["init"])
    end
    pycopy!(pyJuliaError, jl.JuliaError)
    CPyExc_JuliaError[] = incref(getptr(pyJuliaError))
end
