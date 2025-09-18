const INIT_CONSTS_CODE = []

const INIT_MODULES = Dict(
    :pybuiltinsmodule => "builtins",
    :pysysmodule => "sys",
    :pyosmodule => "os",
    :pynumbersmodule => "numbers",
    :pydatetimemodule => "datetime",
    :pycollectionsabcmodule => "collections.abc",
)

for (j, m) in INIT_MODULES
    @eval const $j = pynew()
    push!(INIT_CONSTS_CODE, :(pycopy!($j, pyimport($m))))
end

const INIT_ATTRS = Dict(
    :pyfractiontype => "fractions" => "Fraction",
    :pydatetype => "datetime" => "date",
    :pytimetype => "datetime" => "time",
    :pydatetimetype => "datetime" => "datetime",
    :pytimedeltatype => "datetime" => "timedelta",
)

for (j, k) in INIT_ATTRS
    @eval const $j = pynew()
    push!(INIT_CONSTS_CODE, :(pycopy!($j, pyimport($k))))
end

const BUILTINS = Set([
    # consts
    :True,
    :False,
    :NotImplemented,
    :None,
    :Ellipsis,
    # classes/functions
    :abs,
    :all,
    :any,
    :ascii,
    :bin,
    :bool,
    :bytes,
    :bytearray,
    :callable,
    :chr,
    :classmethod,
    :compile,
    :complex,
    :delattr,
    :dict,
    :dir,
    :divmod,
    :enumerate,
    :eval,
    :exec,
    :filter,
    :float,
    :format,
    :frozenset,
    :getattr,
    :globals,
    :hasattr,
    :hash,
    :help,
    :hex,
    :id,
    :input,
    :int,
    :isinstance,
    :issubclass,
    :iter,
    :len,
    :list,
    :locals,
    :map,
    :max,
    :memoryview,
    :min,
    :next,
    :object,
    :oct,
    :open,
    :ord,
    :pow,
    :print,
    :property,
    :range,
    :repr,
    :reversed,
    :round,
    :set,
    :setattr,
    :slice,
    :sorted,
    :staticmethod,
    :str,
    :sum,
    :super,
    :tuple,
    :type,
    :vars,
    :zip,
    # exceptions
    :BaseException,
    :Exception,
    :StopIteration,
    :GeneratorExit,
    :ArithmeticError,
    :LookupError,
    :AssertionError,
    :AttributeError,
    :BufferError,
    :EOFError,
    :FloatingPointError,
    :OSError,
    :ImportError,
    :IndexError,
    :KeyError,
    :KeyboardInterrupt,
    :MemoryError,
    :NameError,
    :OverflowError,
    :RuntimeError,
    :RecursionError,
    :NotImplementedError,
    :SyntaxError,
    :IndentationError,
    :TabError,
    :ReferenceError,
    :SystemError,
    :SystemExit,
    :TypeError,
    :UnboundLocalError,
    :UnicodeError,
    :UnicodeEncodeError,
    :UnicodeDecodeError,
    :UnicodeTranslateError,
    :ValueError,
    :ZeroDivisionError,
    :BlockingIOError,
    :BrokenPipeError,
    :ChildProcessError,
    :ConnectionError,
    :ConnectionAbortedError,
    :ConnectionRefusedError,
    :FileExistsError,
    :FileNotFoundError,
    :InterruptedError,
    :IsADirectoryError,
    :NotADirectoryError,
    :PermissionError,
    :ProcessLookupError,
    :TimeoutError,
    :EnvironmentError,
    :IOError,
    :Warning,
    :UserWarning,
    :DeprecationWarning,
    :PendingDeprecationWarning,
    :SyntaxWarning,
    :RuntimeWarning,
    :FutureWarning,
    :ImportWarning,
    :UnicodeWarning,
    :BytesWarning,
    :ResourceWarning,
])

# Populate the pybuiltins module imported from API
@eval pybuiltins begin
$([:(const $k = $pynew()) for k in BUILTINS]...)
end

for k in BUILTINS
    if k == :help
        # help is only available in interactive contexts (imported by the 'site' module)
        # see: https://docs.python.org/3/library/functions.html#help
        # see: https://github.com/JuliaPy/PythonCall.jl/issues/248
        push!(
            INIT_CONSTS_CODE,
            :(pycopy!(
                pybuiltins.$k,
                pygetattr(pybuiltinsmodule, $(string(k)), pybuiltins.None),
            )),
        )
    else
        push!(
            INIT_CONSTS_CODE,
            :(pycopy!(pybuiltins.$k, pygetattr(pybuiltinsmodule, $(string(k))))),
        )
    end
end

@eval function init_consts()
    $(INIT_CONSTS_CODE...)
    return
end
