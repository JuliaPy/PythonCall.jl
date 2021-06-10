if DEBUG
    struct PyHdl
        key :: Int
    end
    const PyNULL = PyHdl(0)
else
    struct PyHdl
        ptr :: C.PyPtr
    end
    const PyNULL = PyHdl(C.PyNULL)
end

struct PyAutoHdl{check, close}
    hdl :: PyHdl
end

Base.@kwdef mutable struct Builtins
    bool :: PyHdl = PyNULL
    bytes :: PyHdl = PyNULL
    complex :: PyHdl = PyNULL
    dict :: PyHdl = PyNULL
    enumerate :: PyHdl = PyNULL
    float :: PyHdl = PyNULL
    int :: PyHdl = PyNULL
    list :: PyHdl = PyNULL
    object :: PyHdl = PyNULL
    print :: PyHdl = PyNULL
    set :: PyHdl = PyNULL
    frozenset :: PyHdl = PyNULL
    slice :: PyHdl = PyNULL
    super :: PyHdl = PyNULL
    str :: PyHdl = PyNULL
    tuple :: PyHdl = PyNULL
    type :: PyHdl = PyNULL
    True :: PyHdl = PyNULL
    False :: PyHdl = PyNULL
    NotImplemented :: PyHdl = PyNULL
    None :: PyHdl = PyNULL
    Ellipsis :: PyHdl = PyNULL
    BaseException :: PyHdl = PyNULL
    Exception :: PyHdl = PyNULL
    StopIteration :: PyHdl = PyNULL
    GeneratorExit :: PyHdl = PyNULL
    ArithmeticError :: PyHdl = PyNULL
    LookupError :: PyHdl = PyNULL
    AssertionError :: PyHdl = PyNULL
    AttributeError :: PyHdl = PyNULL
    BufferError :: PyHdl = PyNULL
    EOFError :: PyHdl = PyNULL
    FloatingPointError :: PyHdl = PyNULL
    OSError :: PyHdl = PyNULL
    ImportError :: PyHdl = PyNULL
    IndexError :: PyHdl = PyNULL
    KeyError :: PyHdl = PyNULL
    KeyboardInterrupt :: PyHdl = PyNULL
    MemoryError :: PyHdl = PyNULL
    NameError :: PyHdl = PyNULL
    OverflowError :: PyHdl = PyNULL
    RuntimeError :: PyHdl = PyNULL
    RecursionError :: PyHdl = PyNULL
    NotImplementedError :: PyHdl = PyNULL
    SyntaxError :: PyHdl = PyNULL
    IndentationError :: PyHdl = PyNULL
    TabError :: PyHdl = PyNULL
    ReferenceError :: PyHdl = PyNULL
    SystemError :: PyHdl = PyNULL
    SystemExit :: PyHdl = PyNULL
    TypeError :: PyHdl = PyNULL
    UnboundLocalError :: PyHdl = PyNULL
    UnicodeError :: PyHdl = PyNULL
    UnicodeEncodeError :: PyHdl = PyNULL
    UnicodeDecodeError :: PyHdl = PyNULL
    UnicodeTranslateError :: PyHdl = PyNULL
    ValueError :: PyHdl = PyNULL
    ZeroDivisionError :: PyHdl = PyNULL
    BlockingIOError :: PyHdl = PyNULL
    BrokenPipeError :: PyHdl = PyNULL
    ChildProcessError :: PyHdl = PyNULL
    ConnectionError :: PyHdl = PyNULL
    ConnectionAbortedError :: PyHdl = PyNULL
    ConnectionRefusedError :: PyHdl = PyNULL
    FileExistsError :: PyHdl = PyNULL
    FileNotFoundError :: PyHdl = PyNULL
    InterruptedError :: PyHdl = PyNULL
    IsADirectoryError :: PyHdl = PyNULL
    NotADirectoryError :: PyHdl = PyNULL
    PermissionError :: PyHdl = PyNULL
    ProcessLookupError :: PyHdl = PyNULL
    TimeoutError :: PyHdl = PyNULL
    EnvironmentError :: PyHdl = PyNULL
    IOError :: PyHdl = PyNULL
    Warning :: PyHdl = PyNULL
    UserWarning :: PyHdl = PyNULL
    DeprecationWarning :: PyHdl = PyNULL
    PendingDeprecationWarning :: PyHdl = PyNULL
    SyntaxWarning :: PyHdl = PyNULL
    RuntimeWarning :: PyHdl = PyNULL
    FutureWarning :: PyHdl = PyNULL
    ImportWarning :: PyHdl = PyNULL
    UnicodeWarning :: PyHdl = PyNULL
    BytesWarning :: PyHdl = PyNULL
    ResourceWarning :: PyHdl = PyNULL
end

mutable struct Context
    _c :: C.Context
    _builtins :: Builtins

    @static if DEBUG
        _handles :: Dict{Int, C.PyPtr}
        _nextkey :: Int
    end
end

struct Builtin{name}
    ctx :: Context
end
