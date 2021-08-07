__version__ = '#master'

CONFIG = dict()

def init():
    import os, os.path, sys, ctypes as c, types, shutil, subprocess, jill.install as jli

    # Determine the if we are in a virtual environment
    venvprefix = os.environ.get("VIRTUAL_ENV")
    condaprefix = os.environ.get("CONDA_PREFIX")
    if venvprefix and condaprefix:
        raise Exception('You appear to be in both a virtual env and a conda env.')
    elif venvprefix:
        prefix = venvprefix
    elif condaprefix:
        prefix = condaprefix
    else:
        prefix = None
    if prefix is not None:
        jlprefix = os.path.join(prefix, ".julia")
        jlbin = os.path.join(jlprefix, "bin")
        jlinstall = os.path.join(jlprefix, "install")
        jldownload = os.path.join(jlprefix, "download")
        jlenv = os.path.join(jlprefix, "env")

    # Find the Julia library
    libpath = os.environ.get('PYTHON_JULIACALL_LIB')
    if libpath is None:
        # Find the Julia executable
        # TODO: check the version
        exepath = os.environ.get('PYTHON_JULIACALL_EXE')
        if exepath is None and prefix is not None:
            # see if installed in the virtual env
            exepath = shutil.which(os.path.join(jlbin, "julia"))
        if exepath is None:
            # try the path
            exepath = shutil.which('julia')
        if exepath is None:
            # when using juliaup, the 'julia' command is not in the path but still executable
            try:
                subprocess.run(["julia", "--version"], stdout=subprocess.DEVNULL)
                exepath = "julia"
            except:
                pass
        if exepath is None and prefix is not None:
            # install in the virtual env
            os.makedirs(jldownload)
            d = os.getcwd()
            try:
                os.chdir(jldownload)
                jli.install_julia(confirm=True, install_dir=jlinstall, symlink_dir=jlbin)
            finally:
                os.chdir(d)
            exepath = shutil.which(os.path.join(jlbin, "julia"))
            if exepath is None:
                raise Exception('Installed julia in %s but cannot find it' % repr(jlbin))
        if exepath is None:
            raise Exception('Could not find julia.\n- It is recommended to use this package in a virtual environment (or conda environment)\n  so that Julia may be automatically installed.\n- Otherwise, please install Julia and ensure it is in your PATH.')
        # Test the executable is executable
        try:
            subprocess.run([exepath, "--version"], stdout=subprocess.DEVNULL)
        except:
            raise Exception('Julia executable %s does not exist' % repr(exepath))
        CONFIG['exepath'] = exepath
        libpath = subprocess.run([exepath, '--startup-file=no', '-O0', '--compile=min', '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'], stdout=(subprocess.PIPE)).stdout.decode('utf8')
    else:
        if not os.path.isfile(libpath):
            raise Exception('PYTHON_JULIACALL_LIB=%s does not exist' % repr(libpath))
    CONFIG['libpath'] = libpath
    d = os.getcwd()
    try:
        os.chdir(os.path.dirname(libpath))
        lib = c.CDLL(libpath)
        CONFIG['lib'] = lib
        lib.jl_init__threading.argtypes = []
        lib.jl_init__threading.restype = None
        lib.jl_init__threading()
        lib.jl_eval_string.argtypes = [c.c_char_p]
        lib.jl_eval_string.restype = c.c_void_p
        script = '''
            try
                import Pkg
                function install_PythonCall()
                    version_str = "{}"
                    if '#' in version_str
                        version = nothing
                        url, rev = split(version_str, '#', limit=2)
                        if isempty(url)
                            url = "https://github.com/cjdoris/PythonCall.jl"
                        end
                        Pkg.add(url=url, rev=rev)
                    else
                        version = VersionNumber(version_str)
                        uuid = Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d")
                        deps = Pkg.dependencies()
                        if haskey(deps, uuid)
                            dep = deps[uuid]
                            if version == dep.version
                                return
                            end
                        end
                        Pkg.add(name="PythonCall", version=version)
                    end
                    return
                end
                Pkg.activate("{}", shared={}, io=devnull)
                install_PythonCall()
                ENV["JULIA_PYTHONCALL_LIBPTR"] = "{}"
                import PythonCall
            catch err
                @error "Error loading PythonCall.jl" err=err
                rethrow()
            end
            '''.format(
                __version__,
                ('PythonCall' if prefix is None else jlenv).replace('\\', '\\\\'),
                'true' if prefix is None else 'false',
                c.pythonapi._handle,
            )
        res = lib.jl_eval_string(script.encode('utf8'))
        if res is None:
            raise Exception('PythonCall.jl did not start properly.\n- Ensure that the PythonCall package is installed in Julia.')
    finally:
        os.chdir(d)

init()
del init
