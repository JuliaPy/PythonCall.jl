# This module gets modified by PythonCall when it is loaded, e.g. to include Core, Base
# and Main modules.

__version__ = '0.7.1'

def newmodule(name):
    "A new module with the given name."
    return Base.Module(Base.Symbol(name))

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

    # Determine if we should skip initialising.
    CONFIG['noinit'] = os.getenv('PYTHON_JULIACALL_NOINIT', '') == 'yes'
    if CONFIG['noinit']:
        return

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
        CONFIG['lib'] = lib = c.CDLL(libpath)
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
