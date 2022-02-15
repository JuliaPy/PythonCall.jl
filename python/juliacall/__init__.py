__version__ = '0.5.1'

CONFIG = {}

def newmodule(name):
    "A new module with the given name."
    from . import Base
    return Base.Module(Base.Symbol(name))

class As:
    "Interpret 'value' as type 'type' when converting to Julia."
    __slots__ = ("value", "type")
    __module__ = "juliacall"
    def __init__(self, value, type):
        self.value = value
        self.type = type
    def __repr__(self):
        return "juliacall.As({!r}, {!r})".format(self.value, self.type)

class JuliaError(Exception):
    "An error arising in Julia code."
    __module__ = "juliacall"
    def __init__(self, exception, stacktrace=None):
        super().__init__(exception, stacktrace)
    def __str__(self):
        e = self.exception
        if isinstance(e, str):
            return e
        else:
            from . import Base
            io = Base.IOBuffer()
            Base.showerror(io, e)
            return str(Base.String(Base.take_b(io)))
    @property
    def exception(self):
        return self.args[0]
    @property
    def stacktrace(self):
        return self.args[1]

from .init import init
init()
