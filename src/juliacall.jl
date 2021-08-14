const pyjuliacallmodule = pynew()
const pyJuliaError = pynew()

function init_juliacall()
    # ensure the 'juliacall' module exists
    # this means that Python code can do "import juliacall" and it will work regardless of
    # whether Python is embedded in Julia or vice-versa
    jl = pyjuliacallmodule
    sys = pysysmodule
    if C.CTX.is_embedded
        # in this case, Julia is being embedded into Python by juliacall, which already exists
        pycopy!(jl, sys.modules["juliacall"])
        @assert pystr_asstring(jl.CONFIG["meta"]) == Deps.meta_file()
    elseif "juliacall" in sys.modules
        # otherwise, Python is being embedded into Julia by PythonCall, so should not exist
        error("'juliacall' module already exists")
    else
        # create the juliacall module and save it in sys.modules
        pycopy!(jl, pytype(sys)("juliacall"))
        jl.CONFIG = pydict(embedded=true, meta=Deps.meta_file())
        # jl.__version__ = TODO
        jl.__path__ = pylist((joinpath(dirname(dirname(pathof(PythonCall))), "juliacall"),))
        sys.modules["juliacall"] = jl
    end
end

function init_juliacall_2()
    jl = pyjuliacallmodule
    jl.Main = Main
    jl.Core = Core
    jl.Base = Base
    jl.Pkg = Pkg
    if !C.CTX.is_embedded
        pybuiltins.exec(pybuiltins.compile("from .all import *", "$(@__FILE__):$(@__LINE__)", "exec"), jl.__dict__)
    end
    pycopy!(pyJuliaError, jl.JuliaError)
    C.POINTERS.PyExc_JuliaError = incref(getptr(pyJuliaError))
end

function pyconvert_rule_jlas(::Type{T}, x::Py) where {T}
    # get the type
    t = x.type
    if !pyisjl(t)
        pydel!(t)
        return pyconvert_unconverted()
    end
    S = _pyjl_getvalue(t)
    pydel!(t)
    S isa Type || return pyconvert_unconverted()
    # convert x to S, then to T
    r = pytryconvert(S, x)
    if pyconvert_isunconverted(r)
        return pyconvert_unconverted()
    elseif T == Any || S <: T
        return r
    else
        return pyconvert_tryconvert(T, pyconvert_result(r))
    end
end
