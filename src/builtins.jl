const pybuiltins = PyLazyObject(() -> pyimport("builtins"))
export pybuiltins

# help
pyhelp(args...; opts...) = (pybuiltins.help(args...; opts...); nothing)
export pyhelp

pyprint(args...; opts...) = (pybuiltins.print(args...; opts...); nothing)
export pyprint

# types
for p in [:classmethod, :enumerate, :filter, :map, :property, :reversed, :pystaticmethod, :super, :zip]
    j = Symbol(:py, lowercase(string(p)), :type)
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end

# other objects
for p in [
    # functions (note: some of these may become Julia functions in the future, so don't rely on them being Python objects)
    :all, :any, :chr, :compile, :eval, :exec, :format, :hex, :id, :max, :min, :next, :oct, :open, :ord, :round, :sorted, :sum, :vars,
    # singletons
    :Ellipsis, :NotImplemented,
    # exception, error and warning types
    :ArithmeticError, :AssertionError, :AttributeError, :BaseException, :BlockingIOError, :BrokenPipeError, :BufferError, :BytesWarning, :ChildProcessError, :ConnectionAbortedError, :ConnectionError, :ConnectionRefusedError, :ConnectionResetError, :DeprecationWarning, :EOFError, :EnvironmentError, :Exception, :FileExistsError, :FileNotFoundError, :FloatingPointError, :FutureWarning, :GeneratorExit, :IOError, :ImportError, :ImportWarning, :IndentationError, :IndexError, :InterruptedError, :IsADirectoryError, :KeyError, :KeyboardInterrupt, :LookupError, :MemoryError, :ModuleNotFoundError, :NameError, :NotADirectoryError, :NotImplementedError, :OSError, :OverflowError, :PendingDeprecationWarning, :PermissionError, :ProcessLookupError, :RecursionError, :ReferenceError, :ResourceWarning, :RuntimeError, :RuntimeWarning, :StopAsyncIteration, :StopIteration, :SyntaxError, :SyntaxWarning, :SystemError, :SystemExit, :TabError, :TimeoutError, :TypeError, :UnboundLocalError, :UnicodeDecodeError, :UnicodeEncodeError, :UnicodeError, :UnicodeTranslateError, :UnicodeWarning, :UserWarning, :ValueError, :Warning, :WindowsError, :ZeroDivisionError
]
    j = Symbol(:py, lowercase(string(p)))
    @eval const $j = PyLazyObject(() -> pybuiltins.$p)
    @eval export $j
end
