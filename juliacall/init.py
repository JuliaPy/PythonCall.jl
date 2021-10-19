import os, os.path, ctypes as c, shutil, subprocess, jill.install as jli
from . import CONFIG, __version__, deps, jlcompat

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

# Determine where to put the julia environment
# TODO: Can we more direcly figure out the environment from which python was called? Maybe find the first PATH entry containing python?
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
CONFIG['meta'] = os.path.join(jlenv, "PythonCallPyMeta")

# Determine whether or not to skip resolving julia/package versions
skip = deps.can_skip_resolve()

# Find the Julia library, possibly installing Julia
libpath = os.environ.get('PYTHON_JULIACALL_LIB')
if libpath is not None:
    if not os.path.exists(libpath):
        raise ValueError('PYTHON_JULIACALL_LIB={!r} does not exist'.format(libpath))
else:
    # Find the Julia executable
    exepath = os.environ.get('PYTHON_JULIACALL_EXE')
    if exepath is not None:
        v = jlcompat.julia_version_str(exepath)
        if v is None:
            raise ValueError("PYTHON_JULIACALL_EXE={!r} does not exist".format(exepath))
        else:
            CONFIG["exever"] = v
    else:
        compat = deps.required_julia()
        # Default scenario
        if skip:
            # Already know where Julia is
            exepath = skip["jlexe"]
        else:
            # Find the best available version
            exepath = None
            jill_upstream = os.getenv("JILL_UPSTREAM") or "Official"
            exever = deps.best_julia_version(compat, upstream=jill_upstream)
            v = jlcompat.julia_version_str("julia")
            if v is not None and v == exever:
                exepath = "julia"
            elif os.path.isdir(jlbin):
                for f in os.listdir(jlbin):
                    if f.startswith("julia"):
                        x = os.path.join(jlbin, f)
                        v = jlcompat.julia_version_str(x)
                        if v is not None and v == exever:
                            exepath = x
                            break
            # If no such version, install it
            if exepath is None:
                print("Installing Julia version {} to {!r}".format(exever, jlbin))
                os.makedirs(jldownload, exist_ok=True)
                d = os.getcwd()
                p = os.environ.get("PATH")
                try:
                    if p is None:
                        os.environ["PATH"] = jlbin
                    else:
                        os.environ["PATH"] += os.pathsep + jlbin
                    os.chdir(jldownload)
                    jli.install_julia(version=exever, confirm=True, install_dir=jlinstall, symlink_dir=jlbin, upstream=jill_upstream)
                finally:
                    if p is None:
                        del os.environ["PATH"]
                    else:
                        os.environ["PATH"] = p
                    os.chdir(d)
                exepath = os.path.join(jlbin, "julia.cmd" if os.name == "nt" else "julia")
                if not os.path.isfile(exepath):
                    raise Exception('Installed julia in {!r} but cannot find it'.format(jlbin))
        # Check the version is compatible
        v = jlcompat.julia_version_str(exepath)
        assert v is not None and (compat is None or jlcompat.Version(v) in compat)
        CONFIG['exever'] = v
    CONFIG['exepath'] = exepath
    libpath = subprocess.run([exepath, '--startup-file=no', '-O0', '--compile=min', '-e', 'import Libdl; print(abspath(Libdl.dlpath("libjulia")))'], stdout=(subprocess.PIPE)).stdout.decode('utf8')

# Initialize Julia, including installing required packages
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
    if skip:
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
    if not skip:
        deps.record_resolve(pkgs)
finally:
    os.chdir(d)
