# This module gets modified by PythonCall when it is loaded, e.g. to include Core, Base
# and Main modules.

__version__ = '0.9.28'

_newmodule = None

def newmodule(name):
    "A new module with the given name."
    global _newmodule
    if _newmodule is None:
        _newmodule = Main.seval("name -> (n1=Symbol(name); n2=gensym(n1); Main.@eval(module $n2; module $n1; end; end); Main.@eval $n2.$n1)")
    return _newmodule(name)

_convert = None

def convert(T, x):
    "Convert x to a Julia T."
    global _convert
    if _convert is None:
        _convert = PythonCall.JlWrap.seval("pyjlcallback((T,x)->pyjl(pyconvert(pyjlvalue(T)::Type,x)))")
    return _convert(T, x)

def interactive(enable=True):
    "Allow the Julia event loop to run in the background of the Python REPL."
    if enable:
        PythonCall.Compat._set_python_input_hook()
    else:
        PythonCall.Compat._unset_python_input_hook()

class JuliaError(Exception):
    "An error arising in Julia code."
    def __init__(self, exception, backtrace=None):
        super().__init__(exception, backtrace)
    def __str__(self):
        e = self.exception
        b = self.backtrace
        if b is None:
            return Base.sprint(Base.showerror, e)
        else:
            return Base.sprint(Base.showerror, e, b)
    @property
    def exception(self):
        return self.args[0]
    @property
    def backtrace(self):
        return self.args[1]

CONFIG = {'inited': False}

def init():
    import atexit
    import os
    import ctypes as c
    import sys
    import subprocess
    import warnings

    # importing pytorch before juliacall sometimes causes segfaults. TODO: remove
    if "torch" in sys.modules:
        warnings.warn(
            "torch was imported before juliacall. This may cause a segfault. "
            "To avoid this, import juliacall before importing torch. "
            "For updates, see https://github.com/pytorch/pytorch/issues/78829."
        )

    def option(name, default=None, xkey=None, envkey=None):
        """Get an option.

        Options can be set as command line arguments '-X juliacall-{name}={value}' or as
        environment variables 'PYTHON_JULIACALL_{NAME}={value}'.
        """
        k = xkey or 'juliacall-'+name.lower().replace('_', '-')
        v = sys._xoptions.get(k)
        if v is not None:
            return v, f'-X{k}={v}'
        k = envkey or 'PYTHON_JULIACALL_'+name.upper()
        v = os.getenv(k)
        if v is not None:
            return v, f'{k}={v}'
        return default, f'<default>={default}'

    def choice(name, choices, default=None, **kw):
        v, s = option(name, **kw)
        if v is None:
            return default, s
        if v in choices:
            return v, s
        raise ValueError(
            f'{s}: expecting one of {", ".join(choices)}')

    def path_option(name, default=None, check_exists=False, **kw):
        path, s = option(name, **kw)
        if path is not None:
            if check_exists and not os.path.exists(path):
                raise ValueError(f'{s}: path does not exist')
            return os.path.abspath(path), s
        return default, s
    
    def executable_option(name, default=None, **kw):
        import shutil
        _path, s = option(name, **kw)
        if _path is not None:
            path = shutil.which(_path)
            if path is None:
                raise ValueError(f'{s}: executable not found')
            return os.path.abspath(path), s
        return default, s

    def int_option(name, *, accept_auto=False, **kw):
        val, s = option(name, **kw)
        if val is None:
            return None, s
        if accept_auto and val == "auto":
            return "auto", s
        try:
            int(val)
            return val, s
        except ValueError:
            raise ValueError(f'{s}: expecting an int'+(' or auto' if accept_auto else ""))

    def args_from_config(config):
        argv = [config['exepath']]
        for opt, val in config.items():
            if opt.startswith('opt_'):
                if val is None:
                    if opt == 'opt_handle_signals':
                        val = 'no'
                    else:
                        continue
                argv.append('--' + opt[4:].replace('_', '-') + '=' + val)
        argv = [s.encode("utf-8") for s in argv]

        argc = len(argv)
        argc = c.c_int(argc)
        argv = c.POINTER(c.c_char_p)((c.c_char_p * len(argv))(*argv))
        return argc, argv

    # Determine if we should skip initialising.
    CONFIG['init'] = choice('init', ['yes', 'no'], default='yes')[0] == 'yes'
    if not CONFIG['init']:
        return

    # Parse some more options
    CONFIG['opt_home'] = bindir = path_option('home', check_exists=True, envkey='PYTHON_JULIACALL_BINDIR')[0]
    CONFIG['opt_check_bounds'] = choice('check_bounds', ['yes', 'no', 'auto'])[0]
    CONFIG['opt_compile'] = choice('compile', ['yes', 'no', 'all', 'min'])[0]
    CONFIG['opt_compiled_modules'] = choice('compiled_modules', ['yes', 'no'])[0]
    CONFIG['opt_depwarn'] = choice('depwarn', ['yes', 'no', 'error'])[0]
    CONFIG['opt_inline'] = choice('inline', ['yes', 'no'])[0]
    CONFIG['opt_min_optlevel'] = choice('min_optlevel', ['0', '1', '2', '3'])[0]
    CONFIG['opt_optimize'] = choice('optimize', ['0', '1', '2', '3'])[0]
    CONFIG['opt_procs'] = int_option('procs', accept_auto=True)[0]
    CONFIG['opt_sysimage'] = sysimg = path_option('sysimage', check_exists=True)[0]
    CONFIG['opt_threads'] = int_option('threads', accept_auto=True)[0]
    CONFIG['opt_warn_overwrite'] = choice('warn_overwrite', ['yes', 'no'])[0]
    CONFIG['opt_handle_signals'] = choice('handle_signals', ['yes', 'no'])[0]
    CONFIG['opt_startup_file'] = choice('startup_file', ['yes', 'no'])[0]
    CONFIG['opt_heap_size_hint'] = option('heap_size_hint')[0]
    CONFIG['project'] = path_option('project', check_exists=True)[0]
    CONFIG['exepath'] = executable_option('exe')[0]

    # Stop if we already initialised
    if CONFIG['inited']:
        return

    have_exepath = CONFIG['exepath'] is not None
    have_project = CONFIG['project'] is not None
    if have_exepath and have_project:
        pass
    elif (not have_exepath) and (not have_project):
        # we don't import this at the top level because it is not required when
        # juliacall is loaded by PythonCall and won't be available, or if both
        # `exepath` and `project` are set by the user.
        import juliapkg

        # Find the Julia executable and project
        CONFIG['exepath'] = juliapkg.executable()
        CONFIG['project'] = juliapkg.project()
    else:
        raise Exception("Both PYTHON_JULIACALL_PROJECT and PYTHON_JULIACALL_EXE must be set together, not only one of them.")

    exepath = CONFIG['exepath']
    project = CONFIG['project']

    # Find the Julia library
    cmd = [exepath, '--project='+project, '--startup-file=no', '-O0', '--compile=min',
           '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")), "\\0", Sys.BINDIR)']
    libpath, default_bindir = subprocess.run(cmd, check=True, capture_output=True, encoding='utf8').stdout.split('\0')
    assert os.path.exists(libpath)
    assert os.path.exists(default_bindir)
    CONFIG['libpath'] = libpath

    # Add the Julia library directory to the PATH on Windows so Julia's system libraries can
    # be found. They are normally found because they are in the same directory as julia.exe,
    # but python.exe is somewhere else!
    if os.name == 'nt':
        libdir = os.path.dirname(libpath)
        if 'PATH' in os.environ:
            os.environ['PATH'] = libdir + ';' + os.environ['PATH']
        else:
            os.environ['PATH'] = libdir

    # Open the library
    CONFIG['lib'] = lib = c.PyDLL(libpath, mode=c.RTLD_GLOBAL)

    # parse options
    argc, argv = args_from_config(CONFIG)
    jl_parse_opts = lib.jl_parse_opts
    jl_parse_opts.argtypes = [c.c_void_p, c.c_void_p]
    jl_parse_opts.restype = None
    jl_parse_opts(c.pointer(argc), c.pointer(argv))
    assert argc.value == 0

    # initialise julia
    try:
        jl_init = lib.jl_init_with_image__threading
    except AttributeError:
        jl_init = lib.jl_init_with_image
    jl_init.argtypes = [c.c_char_p, c.c_char_p]
    jl_init.restype = None
    jl_init(
        (default_bindir if bindir is None else bindir).encode('utf8'),
        None if sysimg is None else sysimg.encode('utf8'),
    )

    # call jl_atexit_hook() when python exits to gracefully stop the julia runtime,
    # including running finalizers for any objects that still exist
    @atexit.register
    def at_jl_exit():
        jl_atexit_hook = lib.jl_atexit_hook
        jl_atexit_hook.argtypes = [c.c_int]
        jl_atexit_hook.restype = None
        jl_atexit_hook(0)

    # initialise PythonCall
    jl_eval = lib.jl_eval_string
    jl_eval.argtypes = [c.c_char_p]
    jl_eval.restype = c.c_void_p
    def jlstr(x):
        return 'raw"' + x + '"'
    script = '''
    try
        Base.require(Main, :CompilerSupportLibraries_jll)
        import Pkg
        global __PythonCall_libptr = Ptr{{Cvoid}}(UInt({}))
        ENV["JULIA_PYTHONCALL_EXE"] = {}
        Pkg.activate({}, io=devnull)
        using PythonCall
    catch err
        print(stderr, "ERROR: ")
        showerror(stderr, err, catch_backtrace())
        flush(stderr)
        rethrow()
    end
    '''.format(
        hex(c.pythonapi._handle),
        jlstr(sys.executable or ''),
        jlstr(project),
    )
    res = jl_eval(script.encode('utf8'))
    if res is None:
        raise Exception('PythonCall.jl did not start properly')

    CONFIG['inited'] = True

    if CONFIG['opt_handle_signals'] is None:
        if Base.Threads.nthreads() > 1:
            # a warning to help multithreaded users
            # TODO: should we set PYTHON_JULIACALL_HANDLE_SIGNALS=yes whenever PYTHON_JULIACALL_THREADS != 1?
            warnings.warn(
                "Julia was started with multiple threads "
                "but multithreading support is experimental in JuliaCall. "
                "It is recommended to restart Python with the environment variable "
                "PYTHON_JULIACALL_HANDLE_SIGNALS=yes "
                "set, otherwise you may experience segfaults or other crashes. "
                "Note however that this interferes with Python's own signal handling, "
                "so for example Ctrl-C will not raise KeyboardInterrupt. "
                "See https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe? "
                "for further information. "
                "You can suppress this warning by setting "
                "PYTHON_JULIACALL_HANDLE_SIGNALS=no."
            )

    # Next, automatically load the juliacall extension if we are in IPython or Jupyter
    CONFIG['autoload_ipython_extension'] = choice('autoload_ipython_extension', ['yes', 'no'])[0]
    if CONFIG['autoload_ipython_extension'] in {'yes', None}:
        try:
            get_ipython = sys.modules['IPython'].get_ipython

            if CONFIG['autoload_ipython_extension'] is None:
                # Only let the user know if it was not explicitly set
                print(
                    "Detected IPython. Loading juliacall extension. See https://juliapy.github.io/PythonCall.jl/stable/compat/#IPython"
                )

            load_ipython_extension(get_ipython())
        except Exception as e:
            if CONFIG['autoload_ipython_extension'] == 'yes':
                # Only warn if the user explicitly requested the extension to be loaded
                warnings.warn(
                    "Could not load juliacall extension in Jupyter notebook: " + str(e)
                )
            pass


def load_ipython_extension(ip):
    import juliacall.ipython
    juliacall.ipython.load_ipython_extension(ip)

init()
