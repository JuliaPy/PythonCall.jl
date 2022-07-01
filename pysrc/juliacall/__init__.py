# This module gets modified by PythonCall when it is loaded, e.g. to include Core, Base
# and Main modules.
__version__ = '0.9.1'

_newmodule = None

def newmodule(name):
    "A new module with the given name."
    global _newmodule
    if _newmodule is None:
        _newmodule = Main.seval("name -> (n1=Symbol(name); n2=gensym(n1); Main.@eval(module $n2; module $n1; end; end); Main.@eval $n2.$n1)")
    return _newmodule(name)

_convert = None

def convert(T, x):
    "Convert x to a Julia T."
    global _convert
    if _convert is None:
        _convert = PythonCall.seval("pyjlcallback((T,x)->pyjl(pyconvert(pyjlvalue(T)::Type,x)))")
    return _convert(T, x)

class JuliaError(Exception):
    "An error arising in Julia code."
    def __init__(self, exception, backtrace=None):
        super().__init__(exception, backtrace)
    def __str__(self):
        e = self.exception
        b = self.backtrace
        if b is None:
            return Base.sprint(Base.showerror, e)
        else:
            return Base.sprint(Base.showerror, e, b)
    @property
    def exception(self):
        return self.args[0]
    @property
    def backtrace(self):
        return self.args[1]

CONFIG = {'inited': False}

def init():
    from .initialization import initialize
    initialize()

init()
