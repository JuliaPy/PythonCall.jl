"""Julia wrappers initialized by PythonCall."""

import juliacall as jl
from juliacall import JlBase
from .any import JlBase2

# Populated by Julia during initialization.
_jl_methods = {}


class JlIOBase(JlBase2):
    __module__ = "juliacall"
    __slots__ = ()
    def __init__(self, value):
        JlBase.__init__(self, value, jl.Base.IO)
    def __hash__(self):
        return self._jl_callmethod(_jl_methods["__hash__"])
    def close(self):
        return self._jl_callmethod(_jl_methods["close"])
    @property
    def closed(self):
        return self._jl_callmethod(_jl_methods["closed"])
    def fileno(self):
        return self._jl_callmethod(_jl_methods["fileno"])
    def flush(self):
        return self._jl_callmethod(_jl_methods["flush"])
    def isatty(self):
        return self._jl_callmethod(_jl_methods["isatty"])
    def readable(self):
        return self._jl_callmethod(_jl_methods["readable"])
    def readlines(self, hint=-1):
        lines = []
        total = 0
        while hint < 0 or total < hint:
            line = self.readline()
            if line:
                lines.append(line)
                total += len(line)
            else:
                break
        return lines
    def seek(self, offset, whence=0):
        return self._jl_callmethod(_jl_methods["seek"], offset, whence)
    def seekable(self):
        return True
    def tell(self):
        return self._jl_callmethod(_jl_methods["tell"])
    def truncate(self, size=None):
        return self._jl_callmethod(_jl_methods["truncate"], size)
    def writable(self):
        return self._jl_callmethod(_jl_methods["writable"])
    def writelines(self, lines):
        for line in lines:
            self.write(line)
    def __enter__(self):
        return self
    def __exit__(self, t, v, b):
        self.close()
    def __iter__(self):
        return self
    def __next__(self):
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration
class JlBinaryIO(JlIOBase):
    __module__ = "juliacall"
    __slots__ = ()
    def detach(self):
        raise ValueError("Cannot detach '{}'.".format(type(self)))
    def read(self, size=-1):
        return self._jl_callmethod(_jl_methods["read"], size)
    def read1(self, size=-1):
        return self.read(size)
    def readline(self, size=-1):
        return self._jl_callmethod(_jl_methods["readline"], size)
    def readinto(self, b):
        return self._jl_callmethod(_jl_methods["readinto"], b)
    def readinto1(self, b):
        return self.readinto(b)
    def write(self, b):
        return self._jl_callmethod(_jl_methods["write"], b)
class JlTextIO(JlIOBase):
    __module__ = "juliacall"
    __slots__ = ()
    @property
    def encoding(self):
        return "UTF-8"
    @property
    def errors(self):
        return "strict"
    def detach(self):
        raise ValueError("Cannot detach '{}'.".format(type(self)))
    def read(self, size=-1):
        return self._jl_callmethod(_jl_methods["read_2"], size)
    def readline(self, size=-1):
        return self._jl_callmethod(_jl_methods["readline_2"], size)
    def write(self, s):
        return self._jl_callmethod(_jl_methods["write_2"], s)
import io
io.IOBase.register(JlIOBase)
io.BufferedIOBase.register(JlBinaryIO)
io.TextIOBase.register(JlTextIO)
del io
