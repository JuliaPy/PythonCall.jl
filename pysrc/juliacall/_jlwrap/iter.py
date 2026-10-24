"""Julia value wrappers from src/JlWrap/iter.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class IteratorValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    def __iter__(self):
        return self
    def __next__(self):
        return self._jl_callmethod(_jl_methods["__next__"])
