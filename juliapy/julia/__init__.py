_CONFIG = dict()

def _init_():
    import os, os.path, sys, ctypes as c, types, shutil, subprocess
    libpath = os.environ.get('JULIAPY_LIB')
    if libpath is None:
        exepath = os.environ.get('JULIAPY_EXE')
        if exepath is None:
            exepath = shutil.which('julia')
            if exepath is None:
                raise Exception('Cannot find Julia. Ensure it is in your PATH, or set JULIAPY_EXE to its path.')
        else:
            if not os.path.isfile(exepath):
                raise Exception('JULIAPY_EXE=%s does not exist' % repr(exepath))
        _CONFIG['exepath'] = exepath
        libpath = subprocess.run([exepath, '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'], stdout=(subprocess.PIPE)).stdout.decode('utf8')
    else:
        if not os.path.isfile(libpath):
            raise Exception('JULIAPY_LIB=%s does not exist' % repr(libpath))
    _CONFIG['libpath'] = libpath
    try:
        d = os.getcwd()
        os.chdir(os.path.dirname(libpath))
        lib = c.CDLL(libpath)
    finally:
        os.chdir(d)

    _CONFIG['lib'] = lib
    lib.jl_init__threading.argtypes = []
    lib.jl_init__threading.restype = None
    lib.jl_init__threading()
    lib.jl_eval_string.argtypes = [c.c_char_p]
    lib.jl_eval_string.restype = c.c_void_p
    res = lib.jl_eval_string(
        '''
        ENV["PYTHONJL_LIBPTR"] = "{}"
        import Python
        Python.with_gil() do
            Python.pyimport("sys").modules["julia"].Main = Python.pyjl(Main)
        end
        '''.format(c.pythonapi._handle).encode('utf8'))
    if res is None:
        raise Exception('Python.jl did not start properly. Ensure that the Python package is installed in Julia.')

    class Wrapper(types.ModuleType):

        def __getattr__(self, k):
            return getattr(self.Main, k)

        def __dir__(self):
            return super().__dir__() + self.Main.__dir__()

    sys.modules['julia'].__class__ = Wrapper

_init_()
del _init_

Core = Main.Core
Base = Main.Base
Python = Main.Python

def _import(*names):
    Main.eval(Base.Meta.parse('import ' + ', '.join(names)))
