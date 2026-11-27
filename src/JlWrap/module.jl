const pyjlmoduletype = pynew()

function pyjlmodule_dir(self::Module)
    ks = Symbol[]
    append!(ks, names(self, all = true, imported = true))
    for m in ccall(:jl_module_usings, Any, (Any,), self)::Vector
        append!(ks, names(m))
    end
    pylist(pyjl_attr_jl2py(string(k)) for k in ks)
end

function pyjlmodule_seval(self::Module, expr::Py)
    Py(Base.eval(self, Meta.parseall(strip(pyconvert(String, expr)))))
end
# `jl.seval(...)` is often the first call of a JuliaCall session.
precompile(pyjlmodule_seval, (Module, Py))

function pyjlmodule_seval_nogil(self::Module, expr::Py)
    code = pyconvert(String, expr)
    Py(GIL.@unlock Base.eval(self, Meta.parseall(strip(code))))
end
# Packages call `jl.seval(..., nogil=True)` at import time, where compiling it would cost
# much of the time it saves.
precompile(pyjlmodule_seval_nogil, (Module, Py))

function init_module()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.module")
    methods = m._jl_methods
    methods["__dir__"] = pyjl_methodnum(pyjlmodule_dir)
    methods["seval"] = pyjl_methodnum(pyjlmodule_seval)
    methods["seval_nogil"] = pyjl_methodnum(pyjlmodule_seval_nogil)
    pycopy!(pyjlmoduletype, jl.ModuleValue)
end

pyjltype(::Module) = pyjlmoduletype
