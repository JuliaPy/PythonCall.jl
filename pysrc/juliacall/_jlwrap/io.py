"""Julia value wrappers from src/JlWrap/io.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class IOValueBase(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
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
class BinaryIOValue(IOValueBase):
    __module__ = "juliacall"
    __slots__ = ()
    def detach(self):
        raise ValueError("Cannot detach '{}'.".format(type(self)))
    def read(self, size=-1):
        return self._jl_callmethod(_jl_methods["binary_read"], size)
    def read1(self, size=-1):
        return self.read(size)
    def readline(self, size=-1):
        return self._jl_callmethod(_jl_methods["binary_readline"], size)
    def readinto(self, b):
        return self._jl_callmethod(_jl_methods["readinto"], b)
    def readinto1(self, b):
        return self.readinto(b)
    def write(self, b):
        return self._jl_callmethod(_jl_methods["binary_write"], b)
class TextIOValue(IOValueBase):
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
        return self._jl_callmethod(_jl_methods["text_read"], size)
    def readline(self, size=-1):
        return self._jl_callmethod(_jl_methods["text_readline"], size)
    def write(self, s):
        return self._jl_callmethod(_jl_methods["text_write"], s)
import io
io.IOBase.register(IOValueBase)
io.BufferedIOBase.register(BinaryIOValue)
io.TextIOBase.register(TextIOValue)
del io
