import os, os.path, sys, ctypes as c, types, shutil, subprocess, jill.install as jli
from . import CONFIG, __version__
from .deps import get_dep, set_dep

# Determine if this is a development version of juliacall
# i.e. it is installed from the github repo, which contains Project.toml
reporoot = os.path.dirname(os.path.dirname(__file__))
projtoml = os.path.join(reporoot, "Project.toml")
isdev = os.path.isfile(projtoml) and "PythonCall" in open(projtoml, "rb").read().decode("utf8")
CONFIG['dev'] = isdev

# Determine where to put the julia environment
venvprefix = os.environ.get("VIRTUAL_ENV")
condaprefix = os.environ.get("CONDA_PREFIX")
if venvprefix and condaprefix:
    raise Exception("You appear to be using both a virtual environment and a conda environment.")
elif venvprefix:
    prefix = venvprefix
elif condaprefix:
    prefix = condaprefix
else:
    prefix = None
if prefix is None:
    jlenv = "PythonCall"
else:
    jlenv = os.path.join(prefix, "julia_env")
CONFIG['meta'] = os.path.join(jlenv, "PythonCallMeta.toml")

# Determine where to look for julia
jlprefix = os.path.join(os.path.expanduser("~"), ".julia", "pythoncall")
jlbin = os.path.join(jlprefix, "bin")
jlinstall = os.path.join(jlprefix, "install")
jldownload = os.path.join(jlprefix, "download")
jlexe = os.path.join(jlbin, "julia.cmd" if os.name == "nt" else "julia")

# Find the Julia library
libpath = os.environ.get('PYTHON_JULIACALL_LIB')
if libpath is None:
    # Find the Julia executable...
    # ... in a specified location
    exepath = os.environ.get('PYTHON_JULIACALL_EXE')
    # ... in the default prefix
    if exepath is None:
        exepath = shutil.which(jlexe)
    # ... preinstalled
    if exepath is None:
        exepath = shutil.which("julia")
    # ... preinstalled but not in path but still callable somehow (e.g. juliaup)
    if exepath is None:
        try:
            subprocess.run(["julia", "--version"], stdout=subprocess.DEVNULL)
            exepath = "julia"
        except:
            pass
    # ... after installing in the default prefix
    if exepath is None:
        os.makedirs(jldownload, exist_ok=True)
        d = os.getcwd()
        p = os.environ.get("PATH")
        try:
            if p is None:
                os.environ["PATH"] = jlbin
            else:
                os.environ["PATH"] += os.pathsep + jlbin
            os.chdir(jldownload)
            jli.install_julia(confirm=True, install_dir=jlinstall, symlink_dir=jlbin)
        finally:
            if p is None:
                del os.environ["PATH"]
            else:
                os.environ["PATH"] = p
            os.chdir(d)
        exepath = shutil.which(jlexe)
        if exepath is None:
            raise Exception('Installed julia in \'%s\' but cannot find it' % jlbin)
    assert exepath is not None
    # Test the executable is executable
    try:
        subprocess.run([exepath, "--version"], stdout=subprocess.DEVNULL)
    except:
        raise Exception('Julia executable %s does not exist' % repr(exepath))
    # Find the corresponding libjulia
    CONFIG['exepath'] = exepath
    libpath = subprocess.run([exepath, '--startup-file=no', '-O0', '--compile=min', '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'], stdout=(subprocess.PIPE)).stdout.decode('utf8')
else:
    if not os.path.isfile(libpath):
        raise Exception('PYTHON_JULIACALL_LIB=%s does not exist' % repr(libpath))

# Initialize Julia
d = os.getcwd()
try:
    os.chdir(os.path.dirname(libpath))
    lib = c.CDLL(libpath)
    CONFIG['libpath'] = libpath
    CONFIG['lib'] = lib
    lib.jl_init__threading.argtypes = []
    lib.jl_init__threading.restype = None
    lib.jl_init__threading()
    lib.jl_eval_string.argtypes = [c.c_char_p]
    lib.jl_eval_string.restype = c.c_void_p
    if isdev:
        install = 'Pkg.develop(path="{}")'.format(reporoot.replace('\\', '\\\\'))
    elif get_dep('juliacall', 'PythonCall') != __version__:
        install = 'Pkg.add(name="PythonCall", version="{}")'.format(__version__)
    else:
        install = ''
    script = '''
        try
            import Pkg
            Pkg.activate("{}", shared={}, io=devnull)
            {}
            ENV["JULIA_PYTHONCALL_LIBPTR"] = "{}"
            import PythonCall
        catch err
            @error "Error loading PythonCall.jl" err=err
            rethrow()
        end
        '''.format(
            jlenv.replace('\\', '\\\\'),
            'true' if prefix is None else 'false',
            install,
            c.pythonapi._handle,
        )
    res = lib.jl_eval_string(script.encode('utf8'))
    if res is None:
        raise Exception('PythonCall.jl did not start properly')
    if isdev:
        set_dep('juliacall', 'PythonCall', 'dev')
    elif install:
        set_dep('juliacall', 'PythonCall', __version__)
finally:
    os.chdir(d)
