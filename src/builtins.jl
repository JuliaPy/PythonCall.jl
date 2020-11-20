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

# singletons, exceptions and warnings
for p in [
    :Ellipsis, :NotImplemented, :ArithmeticError, :AssertionError, :AttributeError,
    :BaseException, :BlockingIOError, :BrokenPipeError, :BufferError, :BytesWarning,
    :ChildProcessError, :ConnectionAbortedError, :ConnectionError, :ConnectionRefusedError,
    :ConnectionResetError, :DeprecationWarning, :EOFError, :EnvironmentError, :Exception,
    :FileExistsError, :FileNotFoundError, :FloatingPointError, :FutureWarning,
    :GeneratorExit, :IOError, :ImportError, :ImportWarning, :IndentationError, :IndexError,
    :InterruptedError, :IsADirectoryError, :KeyError, :KeyboardInterrupt, :LookupError,
    :MemoryError, :ModuleNotFoundError, :NameError, :NotADirectoryError,
    :NotImplementedError, :OSError, :OverflowError, :PendingDeprecationWarning,
    :PermissionError, :ProcessLookupError, :RecursionError, :ReferenceError,
    :ResourceWarning, :RuntimeError, :RuntimeWarning, :StopAsyncIteration, :StopIteration,
    :SyntaxError, :SyntaxWarning, :SystemError, :SystemExit, :TabError, :TimeoutError,
    :TypeError, :UnboundLocalError, :UnicodeDecodeError, :UnicodeEncodeError, :UnicodeError,
    :UnicodeTranslateError, :UnicodeWarning, :UserWarning, :ValueError, :Warning,
    :WindowsError, :ZeroDivisionError,
]
    j = Symbol(:py, lowercase(string(p)))
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end
