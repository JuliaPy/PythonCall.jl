import os, os.path, ctypes as c, shutil, sys, subprocess
from . import CONFIG, __version__, deps, semver, install

# Determine if this is a development version of juliacall
# i.e. it is installed from the github repo, which contains Project.toml
reporoot = os.path.dirname(os.path.dirname(__file__))

def find_isdev():
    isdev = False
    for n in ["Project.toml", "JuliaProject.toml"]:
        projtoml = os.path.join(reporoot, n)
        if os.path.isfile(projtoml) and "PythonCall" in open(projtoml, "rb").read().decode("utf8"):
            isdev = True
            break
    CONFIG['dev'] = isdev
    return isdev


def find_env_paths():
    # Determine where to look for julia
    jldepot = os.environ.get("JULIA_DEPOT_PATH", "").split(";" if os.name == "nt" else ":")[0] or os.path.join(os.path.expanduser("~"), ".julia")
    jlprefix = os.path.join(jldepot, "pythoncall")

    # Determine where to put the julia environment
    # TODO: Can we more direcly figure out the environment from which python was called? Maybe find the first PATH entry containing python?
    prefixes = [os.environ.get('VIRTUAL_ENV'), os.environ.get('CONDA_PREFIX'), os.environ.get('MAMBA_PREFIX')]
    prefixes = [x for x in prefixes if x is not None]
    if len(prefixes) == 0:
        prefix = None
    elif len(prefixes) == 1:
        prefix = prefixes[0]
    else:
        raise Exception('You are using some mix of virtual, conda and mamba environments, cannot figure out which to use!')
    if prefix is None:
        jlenv = os.path.join(jldepot, "environments", "PythonCall")
    else:
        jlenv = os.path.join(prefix, "julia_env")
    CONFIG['jlenv'] = os.path.join(jlenv)
    CONFIG['meta'] = os.path.join(jlenv, ".PythonCallJuliaCallMeta")
    return jlenv, jlprefix, CONFIG['meta']



def libpath_from_exepath(exepath):
    return subprocess.run(
        [exepath, '--startup-file=no', '-O0', '--compile=min', '-e',
         'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'],
        check=True, stdout=subprocess.PIPE).stdout.decode('utf8')


def find_executable(skip, jlprefix):
    # Find the Julia executable
    exepath = os.environ.get('PYTHON_JULIACALL_EXE', '@auto')
    if exepath.startswith('@'):
        if exepath not in ('@auto', '@system', '@best'):
            raise ValueError(f"PYTHON_JULIACALL_EXE={exepath!r} is not valid (can be @auto, @system or @best)")
        method = exepath[1:]
        exepath = None
        compat = deps.required_julia()
        # Default scenario
        if skip:
            # Already know where Julia is
            exepath = skip["jlexe"]
        else:
            # @auto and @system try the system julia and succeed if it matches the compat bound
            if exepath is None and method in ('auto', 'system'):
                x = shutil.which('julia')
                if x is not None:
                    v = deps.julia_version_str(x)
                    if compat is None or semver.Version(v) in compat:
                        print(f'Found Julia v{v} at {x!r}')
                        exepath = x
                    else:
                        print(f'Incompatible Julia v{v} at {x!r}, require: {compat.jlstr()}')
            # @auto and @best look for the best available version
            if exepath is None and method in ('auto', 'system'):
                exever, exeverinfo = install.best_julia_version(compat)
                default_exeprefix = os.path.join(jlprefix, 'julia-'+exever)
                x = os.path.join(default_exeprefix, 'bin', 'julia.exe' if os.name=='nt' else 'julia')
                v = deps.julia_version_str(x)
                if v is not None and v == exever:
                    print(f'Found Julia v{v} at {x!r}')
                    exepath = x
                elif v is not None:
                    print(f'Incompatible Julia v{v} at {x!r}, require: = {exever}')
                # If no such version, install it
                if exepath is None:
                    install.install_julia(exeverinfo, default_exeprefix)
                    exepath = x
                    if not os.path.isfile(exepath):
                        raise Exception(f'Installed Julia in {default_exeprefix!r} but cannot find it')
        # Failed to find Julia
        if exepath is None:
            raise Exception('Could not find a compatible version of Julia')
        # Check the version is compatible
        v = deps.julia_version_str(exepath)
        assert v is not None and (compat is None or semver.Version(v) in compat)
        CONFIG['exever'] = v
    else:
        v = deps.julia_version_str(exepath)
        if v is None:
            raise ValueError(f"PYTHON_JULIACALL_EXE={exepath!r} is not a Julia executable")
        else:
            CONFIG["exever"] = v
    assert exepath is not None
    CONFIG['exepath'] = exepath
    return exepath


# Find the Julia library
def find_julia_library(skip, jlprefix):
    libpath = os.environ.get('PYTHON_JULIACALL_LIB')
    if libpath is not None:
        if not os.path.exists(libpath):
            raise ValueError(f'PYTHON_JULIACALL_LIB={libpath!r} does not exist')
    else:
        exepath = find_executable(skip, jlprefix)
        libpath = libpath_from_exepath(exepath)
        return libpath


def init_libjulia(libpath):
    current_dir = os.getcwd()
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
    finally:
        os.chdir(current_dir)
    return lib


def install_packages(lib, skip, isdev, jlenv, meta_path):
    if skip:
        install = ''
    else:
        # get required packages
        pkgs = deps.required_packages()
        # add PythonCall
        if isdev:
            pkgs.append(deps.PackageSpec(name="PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d", path=reporoot, dev=True))
        else:
            pkgs.append(deps.PackageSpec(name="PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d", version="= "+__version__))
        # check if pkgs has changed at all
        meta = deps.load_meta(meta_path)
        prev_pkgs = None if meta is None else meta.get('pkgs')
        if prev_pkgs is not None and sorted(prev_pkgs, key=lambda p: p['name']) == sorted([p.dict() for p in pkgs], key=lambda p: p['name']):
            install = ''
        else:
            # Write a Project.toml specifying UUIDs and compatibility of required packages
            if os.path.exists(jlenv):
                shutil.rmtree(jlenv)
            os.makedirs(jlenv)
            with open(os.path.join(jlenv, "Project.toml"), "wt") as fp:
                print('[deps]', file=fp)
                for pkg in pkgs:
                    print(f'{pkg.name} = "{pkg.uuid}"', file=fp)
                print(file=fp)
                print('[compat]', file=fp)
                for pkg in pkgs:
                    if pkg.version:
                        print(f'{pkg.name} = "{pkg.version}"', file=fp)
                print(file=fp)
            # Create install command
            dev_pkgs = [pkg.jlstr() for pkg in pkgs if pkg.dev]
            add_pkgs = [pkg.jlstr() for pkg in pkgs if not pkg.dev]
            if dev_pkgs and add_pkgs:
                install = 'Pkg.develop([{}]); Pkg.add([{}])'.format(', '.join(dev_pkgs), ', '.join(add_pkgs))
            elif dev_pkgs:
                install = 'Pkg.develop([{}])'.format(', '.join(dev_pkgs))
            elif add_pkgs:
                install = 'Pkg.add([{}])'.format(', '.join(add_pkgs))
            else:
                install = ''
    os.environ['JULIA_PYTHONCALL_LIBPTR'] = str(c.pythonapi._handle)
    os.environ['JULIA_PYTHONCALL_EXE'] = sys.executable or ''
    script = f'''
        try
            if ENV["JULIA_PYTHONCALL_EXE"] != "" && get(ENV, "PYTHON", "") == ""
                # Ensures PyCall uses the same Python executable
                ENV["PYTHON"] = ENV["JULIA_PYTHONCALL_EXE"]
            end
            import Pkg
            Pkg.activate(raw"{jlenv}", io=devnull)
            {install}
            import PythonCall
        catch err
            print(stdout, "ERROR: ")
            showerror(stdout, err, catch_backtrace())
            flush(stdout)
            rethrow()
        end
        '''
    res = lib.jl_eval_string(script.encode('utf8'))
    if res is None:
        raise Exception('PythonCall.jl did not start properly')
    if not skip:
        deps.record_resolve(meta_path, pkgs)


def default_init():
    """
    Initialize Julia and its packages.

    After importing `juliacall`, call this method initialize the Julia
    dependencies. This includes first locating the julia executable and library,
    and if none is found, downloading a Julia distribution. Then the required
    Julia packages are installed.
    """
    # Are we developing PythonCall
    isdev = find_isdev()
    # Determine paths to mutable data maintained by PythonCall, and julia
    jlenv, jlprefix, meta_path = find_env_paths()
    # Determine whether or not to skip resolving julia/package versions
    skip = deps.can_skip_resolve(isdev, meta_path)
    # Find a compatible jula executable and libjulia
    libpath = find_julia_library(skip, jlprefix)
    # Initialze libjulia
    lib = init_libjulia(libpath)
    install_packages(lib, skip, isdev, jlenv, meta_path)
