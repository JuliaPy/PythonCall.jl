import os
import sys
import typing
import subprocess
import ctypes

if typing.TYPE_CHECKING:
    CheckPass = bool
    ExpectInfo = str
    ArgInput = str
    Predicate = typing.Callable[[ArgInput], typing.Tuple[bool, ExpectInfo]]
    TransformArg = typing.Callable[[str], str]

julia_info_query = r"""
import Libdl
println(Base.Sys.BINDIR)
println(abspath(Libdl.dlpath("libjulia")))
println(unsafe_string(Base.JLOptions().image_file))
"""

def choice(*xs) -> 'Predicate':
    """Return a predicate to check if the input string is one of the given choices.
    If a choice is not a 'str', we try conversion and test equality.
    """
    def check(value: "str"):
        def test(x):
            t = type(x)
            try:
                return t(value) == x
            except ValueError:
                return False

        try:
            checked_result = any(map(test, xs))
        except ValueError:
            checked_result = False

        expectation = "one of {}".format(", ".join(map(repr, xs)))
        return checked_result, expectation

    return check


def either(a: 'Predicate', b: 'Predicate') -> 'Predicate':
    """Return a new predicate to check either the predicate 'a' or 'b' checks.
    """
    def check(value: "str"):
        is_ok_a, expect_a = a(value)
        is_ok_b, expect_b = b(value)
        return is_ok_a or is_ok_b, "{} or {}".format(expect_a, expect_b)

    return check


def path_exist() -> 'Predicate':
    """Return a predicate to check if the input string is an existing path.
    """
    def check(value: str):
        return os.path.exists(value), "path {} shall exist".format(repr(value))

    return check


def has_type(t: type) -> 'Predicate':
    """return a predicate to check if the input string can be converted to the given type.
    """
    def check(value: str):
        try:
            checked_result = t(value)
        except ValueError:
            checked_result = False
        return checked_result, t.__name__

    return check


def is_value(exact_val) -> 'Predicate':
    """Return a predicate to check if the input string can be converted to the given value.
    """
    def check(value: str):
        t = type(exact_val)
        try:
            checked_result = t(value) == exact_val
        except ValueError:
            checked_result = False
        return checked_result, repr(exact_val)

    return check


def _get_option(option_name: 'str'):
    """Get the option 'option_name' from 'sys._xoptions' or environment variables.
    Options can be set as command line arguments '-X juliacall-{option_name}={value}' or as
    environment variables 'PYTHON_JULIACALL_{OPTION_NAME}={value}'.
    """
    key_cli = "juliacall-" + option_name.lower()
    raw_opt = sys._xoptions.get(key_cli)  # type: str | None
    set_by = None
    if raw_opt:
        set_by = f"set by Python CLI argument {key_cli}"
        return raw_opt, set_by

    key_env = "PYTHON_JULIACALL_" + option_name.upper()
    raw_opt = os.getenv(key_env)
    if raw_opt:
        set_by = f"set by environment variable {key_env}"
        return raw_opt, set_by
    return None, "default"

def config(*,
            aliases: 'typing.Iterable[str]' = (), 
            validator: 'Predicate',
            transform_if_set: 'None | TransformArg' = None):

    def apply(name) -> 'str | None':
        names = [name, *aliases]
        opt_name = "|".join(names)

        for name in names:
            opt, set_by = _get_option(name)
            if opt:
                is_ok, expect = validator(opt)
                if not is_ok:
                    raise ValueError(f"{opt_name}: {expect} ({set_by})")
                
                if transform_if_set:
                    opt = transform_if_set(opt)
                return opt
    return apply

_initialized = False

configurations = dict(
    bindir = config(aliases=["home"], validator=path_exist(), transform_if_set=os.path.abspath),
    sysimage = config(validator=path_exist(), transform_if_set=os.path.abspath),
    check_bounds = config(validator=choice("yes", "no", "auto")),
    compiled_modules = config(validator=choice("yes", "no")),
    depwarn = config(validator=choice("yes", "no", "error")),
    inline = config(validator=choice("yes", "no")),
    min_optlevel = config(validator=choice("0", "1", "2", "3")),
    procs = config(validator=either(has_type(int), is_value("auto"))),
    threads = config(validator=either(has_type(int), is_value("auto"))),
    warn_overwrite = config(validator=choice("yes", "no")),
    init_jl = config(validator=path_exist(), transform_if_set=os.path.abspath)
)

def args_from_config(exepath: str, config: 'dict[str, str | None]'):
    argv = ["--" + opt.replace("_", "-") + "=" + val for opt, val in config.items() if val]
    argv = [exepath] + argv
    
    # python 2 is deprecated, we just consider python 3
    argv = [arg.encode('utf-8') for arg in argv] 

    argc = len(argv)
    argc = ctypes.c_int(argc)
    argv = ctypes.POINTER(ctypes.c_char_p)((ctypes.c_char_p * len(argv))(*argv))  # type: ignore
    return argc, argv


def initialize():
    global _initialized
    
    if _initialized:
        return
    
    init = config(validator=choice("yes", "no"))('init')
    if not init:
        init = 'yes'

    if init == 'no':
        return
    
    # read settings
    settings = { k: v(k) for k, v in configurations.items() }

    # we don't import this at the top level because it is not required when juliacall is
    # loaded by PythonCall and won't be available
    juliapkg = __import__('juliapkg')

    # Find the Julia executable and project
    exepath = juliapkg.executable()
    settings['project'] = project = juliapkg.project()

    # Find the Julia library
    cmd = [exepath, '--project='+project, '--startup-file=no', '-O0', '--compile=min', '-e', julia_info_query]
    default_bindir, libpath, default_sysimage = subprocess.run(
        cmd, check=True, capture_output=True, encoding='utf8'
    ).stdout.splitlines()
    
    if not settings['bindir']:
        settings['bindir'] = default_bindir
    if not settings['sysimage']:
        settings['sysimage'] = default_sysimage

    d = os.getcwd()
    try:
        os.chdir(os.path.dirname(libpath))
        lib = ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
        try:
            init_func = lib.jl_init_with_image
        except AttributeError:
            init_func = lib.jl_init_with_image__threading

        init_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        init_func.restype = None
        
        # these two options must be set as a str
        bindir = settings.pop("bindir")
        sysimage = settings.pop("sysimage")
        init_jl = settings.pop('init_jl') or os.path.join(os.path.dirname(__file__), 'init.jl')

        argc, argv = args_from_config(exepath, settings)
        lib.jl_parse_opts(ctypes.pointer(argc), ctypes.pointer(argv))
        assert argc.value == 0
        assert bindir
        assert sysimage
        init_func(bindir.encode('utf8'), sysimage.encode('utf8'))
        lib.jl_eval_string.argtypes = [ctypes.c_char_p]
        lib.jl_eval_string.restype = ctypes.c_void_p
        os.environ['JULIA_PYTHONCALL_LIBPTR'] = str(ctypes.pythonapi._handle)
        os.environ['JULIA_PYTHONCALL_EXE'] = sys.executable or ''
        os.environ['JULIA_PYTHONCALL_PROJECT'] = project
        os.environ['JULIA_PYTHONCALL_INIT_JL'] = init_jl

        script = '''
        try
            import Pkg
            Pkg.activate(ENV["JULIA_PYTHONCALL_PROJECT"], io=devnull)
            import PythonCall
            # This uses some internals, but Base._start() gets the state more like Julia
            # is if you call the executable directly, in particular it creates workers when
            # the --procs argument is given.
            push!(Core.ARGS, ENV["JULIA_PYTHONCALL_INIT_JL"])
            Base._start()
            @eval Base PROGRAM_FILE=""
        catch err
            print(stderr, "ERROR: ")
            showerror(stderr, err, catch_backtrace())
            flush(stderr)
            rethrow()
        end
        '''

        res = lib.jl_eval_string(script.encode('utf8'))
        if res is None:
            raise Exception('PythonCall.jl did not start properly')

    finally:
        os.chdir(d)

    _initialized = True
