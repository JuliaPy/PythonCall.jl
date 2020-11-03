const pybuiltins = PyLazyObject(() -> pyimport("builtins"))
export pybuiltins

# help
const pyhelpfunc = PyLazyObject(() -> pybuiltins.help)

pyhelp(args...; opts...) = (pyhelpfunc(args...; opts...); nothing)
export pyhelp

# other objects
for p in [
    # functions
    :min, :max,
    # singletons
    :Ellipsis, :NotImplemented,
    # exceptions, errors and warnings
    :ArithmeticError, :AssertionError, :AttributeError, :BaseException, :BlockingIOError, :BrokenPipeError, :BufferError, :BytesWarning, :ChildProcessError, :ConnectionAbortedError, :ConnectionError, :ConnectionRefusedError, :ConnectionResetError, :DeprecationWarning, :EOFError, :EnvironmentError, :Exception, :FileExistsError, :FileNotFoundError, :FloatingPointError, :FutureWarning, :GeneratorExit, :IOError, :ImportError, :ImportWarning, :IndentationError, :IndexError, :InterruptedError, :IsADirectoryError, :KeyError, :KeyboardInterrupt, :LookupError, :MemoryError, :ModuleNotFoundError, :NameError, :NotADirectoryError, :NotImplementedError, :OSError, :OverflowError, :PendingDeprecationWarning, :PermissionError, :ProcessLookupError, :RecursionError, :ReferenceError, :ResourceWarning, :RuntimeError, :RuntimeWarning, :StopAsyncIteration, :StopIteration, :SyntaxError, :SyntaxWarning, :SystemError, :SystemExit, :TabError, :TimeoutError, :TypeError, :UnboundLocalError, :UnicodeDecodeError, :UnicodeEncodeError, :UnicodeError, :UnicodeTranslateError, :UnicodeWarning, :UserWarning, :ValueError, :Warning, :WindowsError, :ZeroDivisionError
]
    j = Symbol(:py, lowercase(string(p)))
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end
