"""Julia value wrappers from src/JlWrap/module.jl."""

from .any import AnyValue
from juliacall import ValueBase

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class ModuleValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __dir__(self):
        return ValueBase.__dir__(self) + self._jl_callmethod(_jl_methods["__dir__"])
    def seval(self, expr):
        return self._jl_callmethod(_jl_methods["seval"], expr)
