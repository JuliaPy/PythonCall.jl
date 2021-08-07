__version__ = '0.2.1'

CONFIG = dict(embedded=False)

def init():
    import os, os.path, sys, ctypes as c, types, shutil, subprocess, jill.install as jli

    # Determine if this is a development version of juliacall
    # i.e. it is installed from the github repo, which contains Project.toml
    reporoot = os.path.dirname(os.path.dirname(__file__))
    projtoml = os.path.join(reporoot, "Project.toml")
    isdev = os.path.isfile(projtoml) and "PythonCall" in open(projtoml, "rb").read().decode("utf8")
    CONFIG['dev'] = isdev

    # Determine where to look for julia
    venvprefix = os.environ.get("VIRTUAL_ENV")
    condaprefix = os.environ.get("CONDA_PREFIX")
    if venvprefix and condaprefix:
        raise Exception("You appear to be using both a virtual environment and a conda environment.")
    elif venvprefix:
        prefix = venvprefix
    elif condaprefix:
        prefix = condaprefix
    else:
        prefix = os.path.dirname(__file__)
    jlprefix = os.path.join(prefix, "julia")
    jlbin = os.path.join(jlprefix, "bin")
    jlinstall = os.path.join(jlprefix, "install")
    jldownload = os.path.join(jlprefix, "download")
    jlenv = os.path.join(jlprefix, "env")
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
        # ... nowhere!
        if exepath is None:
            raise Exception('Could not find julia.\n- It is recommended to use this package in a virtual environment (or conda environment)\n  so that Julia may be automatically installed.\n- Otherwise, please install Julia and ensure it is in your PATH.')
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
        else:
            install = '''
                (function ()
                    version_str = "{}"
                    version = VersionNumber(version_str)
                    uuid = Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d")
                    deps = Pkg.dependencies()
                    if haskey(deps, uuid)
                        dep = deps[uuid]
                        if version == dep.version && dep.is_tracking_registry && !dep.is_tracking_path && !dep.is_tracking_repo
                            return
                        end
                    end
                    Pkg.add(name="PythonCall", version=version)
                    return
                end)()
            '''.format(__version__)
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
            raise Exception('PythonCall.jl did not start properly.\n- Ensure that the PythonCall package is installed in Julia.')
    finally:
        os.chdir(d)

init()
del init
