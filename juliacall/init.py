import os, os.path, ctypes as c, shutil, subprocess, jill.install as jli
from . import CONFIG, __version__, deps

# Determine if this is a development version of juliacall
# i.e. it is installed from the github repo, which contains Project.toml
reporoot = os.path.dirname(os.path.dirname(__file__))
isdev = False
for n in ["Project.toml", "JuliaProject.toml"]:
    projtoml = os.path.join(reporoot, n)
    if os.path.isfile(projtoml) and "PythonCall" in open(projtoml, "rb").read().decode("utf8"):
        isdev = True
        break
CONFIG['dev'] = isdev

# Determine where to look for julia
jldepot = os.environ.get("JULIA_DEPOT_PATH", "").split(";" if os.name == "nt" else ":")[0] or os.path.join(os.path.expanduser("~"), ".julia")
jlprefix = os.path.join(jldepot, "pythoncall")
jlbin = os.path.join(jlprefix, "bin")
jlinstall = os.path.join(jlprefix, "install")
jldownload = os.path.join(jlprefix, "download")
jlexe = os.path.join(jlbin, "julia.cmd" if os.name == "nt" else "julia")

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
    jlenv = os.path.join(jldepot, "environments", "PythonCall")
else:
    jlenv = os.path.join(prefix, "julia_env")
CONFIG['jlenv'] = os.path.join(jlenv)
CONFIG['meta'] = os.path.join(jlenv, "PythonCallMeta.json")

# Find the Julia library
libpath = os.environ.get('PYTHON_JULIACALL_LIB')
if libpath is None:
    # Find the Julia executable...
    # TODO: Check the Julia executable is compatible with jlcompat
    #       - If Julia is not found, install a compatible one.
    #       - If Julia is found in the default prefix and not compatible, reinstall.
    #       - If Julia is found elsewhere, emit a warning?
    jlcompat = deps.required_julia()
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
    if deps.can_skip_resolve():
        pkgs = None
        install = ''
    else:
        # get required packages
        pkgs = deps.required_packages()
        # add PythonCall
        if isdev:
            pkgs.append(deps.PackageSpec(name="PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d", path=reporoot, dev=True))
        else:
            pkgs.append(deps.PackageSpec(name="PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d", compat="= "+__version__))
        # check if pkgs has changed at all
        prev_pkgs = deps.get_meta("pydeps", "pkgs")
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
                    print('{} = "{}"'.format(pkg.name, pkg.uuid), file=fp)
                print(file=fp)
                print('[compat]', file=fp)
                for pkg in pkgs:
                    if pkg.compat:
                        print('{} = "{}"'.format(pkg.name, pkg.compat), file=fp)
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
    script = '''
        try
            import Pkg
            Pkg.activate(raw"{}", io=devnull)
            {}
            ENV["JULIA_PYTHONCALL_LIBPTR"] = "{}"
            import PythonCall
        catch err
            @error "Error loading PythonCall.jl" err=err
            rethrow()
        end
        '''.format(jlenv, install, c.pythonapi._handle)
    res = lib.jl_eval_string(script.encode('utf8'))
    if res is None:
        raise Exception('PythonCall.jl did not start properly')
    if pkgs is not None:
        deps.record_resolve(pkgs)
finally:
    os.chdir(d)
