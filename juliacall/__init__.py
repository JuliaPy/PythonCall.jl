CONFIG = dict()

def init():
    import os, os.path, sys, ctypes as c, types, shutil, subprocess
    libpath = os.environ.get('PYTHON_JULIACALL_LIB')
    if libpath is None:
        exepath = os.environ.get('PYTHON_JULIACALL_EXE')
        if exepath is None:
            exepath = shutil.which('julia')
            if exepath is None:
                raise Exception('Cannot find Julia. Ensure it is in your PATH, set PYTHON_JULIACALL_EXE to its path, or set PYTHON_JULIACALL_LIB to the path to libjuliacall.')
        else:
            if not os.path.isfile(exepath):
                raise Exception('PYTHON_JULIACALL_EXE=%s does not exist' % repr(exepath))
        CONFIG['exepath'] = exepath
        libpath = subprocess.run([exepath, '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'], stdout=(subprocess.PIPE)).stdout.decode('utf8')
    else:
        if not os.path.isfile(libpath):
            raise Exception('PYTHON_JULIACALL_LIB=%s does not exist' % repr(libpath))
    CONFIG['libpath'] = libpath
    try:
        d = os.getcwd()
        os.chdir(os.path.dirname(libpath))
        lib = c.CDLL(libpath)
    finally:
        os.chdir(d)

    CONFIG['lib'] = lib
    lib.jl_init__threading.argtypes = []
    lib.jl_init__threading.restype = None
    lib.jl_init__threading()
    lib.jl_eval_string.argtypes = [c.c_char_p]
    lib.jl_eval_string.restype = c.c_void_p
    res = lib.jl_eval_string(
        '''
        try
            ENV["JULIA_PYTHONCALL_LIBPTR"] = "{}"
            import PythonCall
        catch err
            @error "Error loading PythonCall.jl" err=err
            rethrow()
        end
        '''.format(c.pythonapi._handle).encode('utf8'))
    if res is None:
        raise Exception('PythonCall.jl did not start properly. Ensure that the PythonCall package is installed in Julia.')

init()
del init
