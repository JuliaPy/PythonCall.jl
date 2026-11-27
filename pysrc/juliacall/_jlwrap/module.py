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
    def seval(self, expr, *, nogil=False):
        '''Evaluate the Julia code `expr` (a string) in this module.

        If `nogil` is true, the GIL is released while the code runs, so other Python
        threads can run meanwhile. WARNING: the code must not interact with Python at all
        without re-acquiring the GIL.
        '''
        return self._jl_callmethod(_jl_methods["seval_nogil" if nogil else "seval"], expr)
