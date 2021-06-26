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
    elseif "juliacall" in sys.modules
        # otherwise, Python is being embedded into Julia by PythonCall, so should not exist
        error("'juliacall' module already exists")
    else
        # create the juliacall module and save it in sys.modules
        pycopy!(jl, pytype(sys)("juliacall"))
        jl.CONFIG = pydict()
        sys.modules["juliacall"] = jl
    end

    # TODO: assign Main, Base and Core julia modules
    # jl.Main = pyjl(Main)
    # jl.Base = pyjl(Base)
    # jl.Core = pyjl(Core)

    # define a few more things
    pybuiltins.exec("""
    def newmodule(name):
        "A new module with the given name."
        return Base.Module(Base.Symbol(name))
    class As:
        "Interpret 'value' as type 'type' when converting to Julia."
        __slots__ = ("value", "type")
        def __init__(self, value, type):
            self.value = value
            self.type = type
        def __repr__(self):
            return "juliacall.As({!r}, {!r})".format(self.value, self.type)
    class JuliaError(Exception):
        "An error arising in Julia code."
        pass
    """, jl.__dict__)
    pycopy!(pyJuliaError, jl.JuliaError)
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
