const pybuiltins = PyLazyObject(() -> pyimport("builtins"))
export pybuiltins

# types
for p in [:classmethod, :enumerate, :filter, :map, :property, :reversed, :staticmethod, :super, :zip]
    j = Symbol(:py, p, :type)
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end

# functions
for p in [:all, :any, :chr, :compile, :eval, :exec, :format, :help, :hex, :id, :max, :min, :next, :oct, :open, :ord, :print, :round, :sorted, :sum, :vars]
    j = Symbol(:py, p, :func)
    jf = Symbol(:py, p)
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    if p in [:help, :print, :exec]
        @eval $jf(args...; opts...) = ($j(args...; opts...); nothing)
    else
        @eval $jf(args...; opts...) = $j(args...; opts...)
    end
    @eval export $jf
end

# singletons
for p in [:Ellipsis, :NotImplemented]
    j = Symbol(:py, lowercase(string(p)))
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end

# exceptions and warnings
# NOTE: We import these directly from C, so that the error indicator is not inadvertantly
# reset when the lazy object is evaluated the first time (otherwise there can be problems
# using `pyerroccurred(t)`).
for n in [
    # exceptions
    :BaseException, :Exception, :ArithmeticError, :AssertionError, :AttributeError,
    :BlockingIOError, :BrokenPipeError, :BufferError, :ChildProcessError,
    :ConnectionAbortedError, :ConnectionError, :ConnectionRefusedError,
    :ConnectionResetError, :EOFError, :FileExistsError, :FileNotFoundError,
    :FloatingPointError, :GeneratorExit, :ImportError, :IndentationError, :IndexError,
    :InterruptedError, :IsADirectoryError, :KeyError, :KeyboardInterrupt, :LookupError,
    :MemoryError, :ModuleNotFoundError, :NameError, :NotADirectoryError,
    :NotImplementedError, :OSError, :OverflowError, :PermissionError, :ProcessLookupError,
    :RecursionError, :ReferenceError, :RuntimeError, :StopAsyncIteration, :StopIteration,
    :SyntaxError, :SystemError, :SystemExit, :TabError, :TimeoutError, :TypeError,
    :UnboundLocalError, :UnicodeDecodeError, :UnicodeEncodeError, :UnicodeError,
    :UnicodeTranslateError, :ValueError, :ZeroDivisionError,
    # aliases
    :EnvironmentError, :IOError, :WindowsError,
    # warnings
    :Warning, :BytesWarning, :DeprecationWarning, :FutureWarning, :ImportWarning,
    :PendingDeprecationWarning, :ResourceWarning, :RuntimeWarning, :SyntaxWarning,
    :UnicodeWarning, :UserWarning,
]
    j = Symbol(:py, lowercase(string(n)))
    p = QuoteNode(Symbol(:PyExc_, n))
    @eval const $j = PyLazyObject(() -> pynewobject(unsafe_load(Ptr{CPyPtr}(C.@pyglobal($p))), true))
    @eval export $j
end
