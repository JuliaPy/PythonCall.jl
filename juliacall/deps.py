from . import CONFIG

def load_meta():
    import toml, os.path
    fn = CONFIG['meta']
    if os.path.exists(fn):
        with open(fn) as fp:
            return toml.load(fp)
    else:
        return {}

def save_meta(meta):
    import toml
    fn = CONFIG['meta']
    with open(fn, 'w') as fp:
        toml.dump(meta, fp)

def get_meta(*keys):
    meta = load_meta()
    for key in keys:
        if key in meta:
            meta = meta[key]
        else:
            return None
    return meta

def set_meta(*args):
    if len(args) < 2:
        raise TypeError('set_meta requires at least two arguments')
    keys = args[:-1]
    value = args[-1]
    meta2 = meta = load_meta()
    for key in keys[:-1]:
        meta2 = meta2.setdefault(key, {})
    meta2[keys[-1]] = value
    save_meta(meta)

def get_dep(package, name):
    return get_meta('pydeps', package, name)

def set_dep(package, name, value):
    set_meta('pydeps', package, name, value)

def require(package, name, value, func=None):
    if func is None:
        return Require(package, name, value)
    elif get_dep(package, name) != value:
        func()
        set_dep(package, name, value)

class Require:
    def __init__(self, package, name, value):
        self.package = package
        self.name = name
        self.value = value
    def __enter__(self):
        self.required = get_dep(self.package, self.name) != self.value
        return self.required
    def __exit__(self, t, v, b):
        if t is None and self.required:
            set_dep(self.package, self.name, self.value)

def require_julia(package, name, version):
    with require(package, name, version) as required:
        if required:
            from juliacall import Pkg
            Pkg.add(name=name, version=version)
