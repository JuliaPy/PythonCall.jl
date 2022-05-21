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
    Py(Base.eval(self, Meta.parse(pyconvert(String, expr))))
end

function init_jlwrap_module()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class ModuleValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
        def __dir__(self):
            return ValueBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlmodule_dir)))
        def seval(self, expr):
            return self._jl_callmethod($(pyjl_methodnum(pyjlmodule_seval)), expr)
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlmoduletype, jl.ModuleValue)
end

pyjltype(::Module) = pyjlmoduletype
