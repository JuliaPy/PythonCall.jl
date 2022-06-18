# This module gets modified by PythonCall when it is loaded, e.g. to include Core, Base
# and Main modules.

__version__ = '0.9.0'

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
        _convert = PythonCall.seval("pyjlcallback((T,x)->pyjl(pyconvert(pyjlvalue(T)::Type,x)))")
    return _convert(T, x)

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
    import os
    import ctypes as c
    import sys
    import subprocess

    def option(name, default=None):
        """Get an option.

        Options can be set as command line arguments '-X juliacall_{name}={value}' or as
        environment variables 'PYTHON_JULIACALL_{NAME}={value}'.
        """
        k = 'juliacall_'+name.lower()
        v = sys._xoptions.get(k)
        if v is not None:
            return v
        k = 'PYTHON_JULIACALL_'+name.upper()
        v = os.getenv(k)
        if v is not None:
            return v
        return default

    def choice(name, choices, default=None):
        k = option(name)
        if k is None:
            return default
        if k in choices:
            if isinstance(k, dict):
                return choices[k]
            else:
                return k
        raise ValueError(f'invalid value for option: JULIACALL_{name.upper()}={k}, expecting one of {", ".join(choices)}')

    def path_option(name, default=None):
        path = option(name)
        if path is not None:
            return os.path.abspath(path)
        return default

    # Determine if we should skip initialising.
    CONFIG['init'] = choice('init', ['yes', 'no'], default='yes') == 'yes'
    if not CONFIG['init']:
        return

    # Parse some more options
    CONFIG['opt_bindir'] = path_option('bindir')  # TODO
    CONFIG['opt_check_bounds'] = choice('check_bounds', ['yes', 'no'])  # TODO
    CONFIG['opt_compile'] = choice('compile', ['yes', 'no', 'all', 'min'])  # TODO
    CONFIG['opt_compiled_modules'] = choice('compiled_modules', ['yes', 'no'])  # TODO
    CONFIG['opt_depwarn'] = choice('depwarn', ['yes', 'no', 'error'])  # TODO
    CONFIG['opt_inline'] = choice('inline', ['yes', 'no'])  # TODO
    CONFIG['opt_optimize'] = choice('optimize', ['0', '1', '2', '3'])  # TODO
    CONFIG['opt_sysimage'] = path_option('sysimage')  # TODO
    CONFIG['opt_warn_overwrite'] = choice('warn_overwrite', ['yes', 'no'])  # TODO

    # Stop if we already initialised
    if CONFIG['inited']:
        return

    # we don't import this at the top level because it is not required when juliacall is
    # loaded by PythonCall and won't be available
    import juliapkg

    # Find the Julia executable and project
    CONFIG['exepath'] = exepath = juliapkg.executable()
    CONFIG['project'] = project = juliapkg.project()

    # Find the Julia library
    cmd = [exepath, '--project='+project, '--startup-file=no', '-O0', '--compile=min', '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))']
    CONFIG['libpath'] = libpath = subprocess.run(cmd, check=True, capture_output=True, encoding='utf8').stdout
    assert os.path.exists(libpath)

    # Initialise Julia
    d = os.getcwd()
    try:
        # Open the library
        os.chdir(os.path.dirname(libpath))
        CONFIG['lib'] = lib = c.CDLL(libpath, mode=c.RTLD_GLOBAL)
        lib.jl_init__threading.argtypes = []
        lib.jl_init__threading.restype = None
        lib.jl_init__threading()
        lib.jl_eval_string.argtypes = [c.c_char_p]
        lib.jl_eval_string.restype = c.c_void_p
        os.environ['JULIA_PYTHONCALL_LIBPTR'] = str(c.pythonapi._handle)
        os.environ['JULIA_PYTHONCALL_EXE'] = sys.executable or ''
        os.environ['JULIA_PYTHONCALL_PROJECT'] = project
        script = '''
            try
                import Pkg
                Pkg.activate(ENV["JULIA_PYTHONCALL_PROJECT"], io=devnull)
                import PythonCall
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

    CONFIG['inited'] = True

init()
