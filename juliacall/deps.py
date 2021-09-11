import os
import sys

from time import time

from . import CONFIG, __version__
from .jlcompat import JuliaCompat, Version

### META

def load_meta():
    import json, os.path
    fn = CONFIG['meta']
    if os.path.exists(fn):
        with open(fn) as fp:
            return json.load(fp)
    else:
        return {}

def save_meta(meta):
    import json
    fn = CONFIG['meta']
    with open(fn, 'w') as fp:
        json.dump(meta, fp)

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

### RESOLVE

class PackageSpec:
    def __init__(self, name, uuid, dev=False, compat=None, path=None, url=None, rev=None, version=None):
        self.name = name
        self.uuid = uuid
        self.dev = dev
        self.compat = compat
        self.path = path
        self.url = url
        self.rev = rev
        self.version = version

    def jlstr(self):
        args = ['name="{}"'.format(self.name), 'uuid="{}"'.format(self.uuid)]
        if self.path is not None:
            args.append('path=raw"{}"'.format(self.path))
        if self.url is not None:
            args.append('url=raw"{}"'.format(self.url))
        if self.rev is not None:
            args.append('rev=raw"{}"'.format(self.rev))
        if self.version is not None:
            args.append('version=raw"{}"'.format(self.version))
        return "Pkg.PackageSpec({})".format(', '.join(args))

    def dict(self):
        ans = {
            "name": self.name,
            "uuid": self.uuid,
            "dev": self.dev,
            "compat": self.compat,
            "path": self.path,
            "url": self.url,
            "rev": self.rev,
            "version": self.version,
        }
        return {k:v for (k,v) in ans.items() if v is not None}

def can_skip_resolve():
    # resolve if we haven't resolved before
    deps = get_meta("pydeps")
    if deps is None:
        return False
    # resolve whenever the version changes
    version = deps.get("version")
    if version is None or version != __version__:
        return False
    # resolve whenever swapping between dev/not dev
    isdev = deps.get("dev")
    if isdev is None or isdev != CONFIG["dev"]:
        return False
    # resolve whenever anything in sys.path changes
    timestamp = deps.get("timestamp")
    if timestamp is None:
        return False
    sys_path = deps.get("sys_path")
    if sys_path is None or sys_path != sys.path:
        return False
    for path in sys.path:
        if not path:
            path = os.getcwd()
        if not os.path.exists(path):
            continue
        if os.path.getmtime(path) > timestamp:
            return False
        if os.path.isdir(path):
            fn = os.path.join(path, "juliacalldeps.json")
            if os.path.exists(fn) and os.path.getmtime(fn) > timestamp:
                return False
    return True

def deps_files():
    ans = []
    for path in sys.path:
        if not path:
            path = os.getcwd()
        if not os.path.isdir(path):
            continue
        fn = os.path.join(path, "juliacalldeps.json")
        if os.path.isfile(fn):
            ans.append(fn)
        for subdir in os.listdir(path):
            fn = os.path.join(path, subdir, "juliacalldeps.json")
            if os.path.isfile(fn):
                ans.append(fn)
    return list(set(ans))

def required_packages():
    # read all dependencies into a dict: name -> key -> file -> value
    import json
    all_deps = {}
    for fn in deps_files():
        with open(fn) as fp:
            deps = json.load(fp)
        for (name, kvs) in deps.get("packages", {}).items():
            dep = all_deps.setdefault(name, {})
            for (k, v) in kvs.items():
                dep.setdefault(k, {})[fn] = v
    # merges non-unique values
    def merge_unique(dep, kfvs, k):
        fvs = kfvs.pop(k, None)
        if fvs is not None:
            vs = set(fvs.values())
            if len(vs) == 1:
                dep[k], = vs
            elif vs:
                raise Exception("'{}' entries are not unique:\n{}".format(k, '\n'.join(['- {!r} at {}'.format(v,f) for (f,v) in fvs.items()])))
    # merges compat entries
    def merge_compat(dep, kfvs, k):
        fvs = kfvs.pop(k, None)
        if fvs is not None:
            compats = list(map(JuliaCompat, fvs.values()))
            compat = compats[0]
            for c in compats[1:]:
                compat &= c
            if compat.isempty():
                raise Exception("'{}' entries have empty intersection:\n{}".format(k, '\n'.join(['- {!r} at {}'.format(v,f) for (f,v) in fvs.items()])))
            else:
                dep[k] = compat.jlstr()
    # merges booleans with any
    def merge_any(dep, kfvs, k):
        fvs = kfvs.pop(k, None)
        if fvs is not None:
            dep[k] = any(fvs.values())
    # merge dependencies: name -> key -> value
    deps = []
    for (name, kfvs) in all_deps.items():
        kw = {'name': name}
        merge_unique(kw, kfvs, 'uuid')
        merge_unique(kw, kfvs, 'path')
        merge_unique(kw, kfvs, 'url')
        merge_unique(kw, kfvs, 'rev')
        merge_compat(kw, kfvs, 'compat')
        merge_any(kw, kfvs, 'dev')
        deps.append(PackageSpec(**kw))
    return deps

def required_julia():
    import json
    compats = {}
    for fn in deps_files():
        with open(fn) as fp:
            deps = json.load(fp)
            c = deps.get("julia")
            if c is not None:
                compats[fn] = JuliaCompat(c)
    compat = None
    for c in compats.values():
        if compat is None:
            compat = c
        else:
            compat &= c
    if compat is not None and compat.isempty():
        raise Exception("'julia' compat entries have empty intersection:\n{}".format('\n'.join(['- {!r} at {}'.format(v,f) for (f,v) in compats.items()])))
    return compat

def best_julia_version():
    """
    Selects the best Julia version available matching required_julia().

    It's based on jill.utils.version_utils.latest_version() and jill.install.install_julia().
    """
    import jill.utils.version_utils
    import jill.install
    compat = required_julia()
    system = jill.install.current_system()
    arch = jill.install.current_architecture()
    if system == 'linux' and jill.install.current_libc() == 'musl':
        system = 'musl'
    releases = jill.utils.version_utils.read_releases()
    releases = [r for r in releases if r[1]==system and r[2]==arch]
    if compat is not None:
        _releases = releases
        releases = []
        for r in _releases:
            try:
                v = Version(r[0])
                if v in compat:
                    releases.append(r)
            except:
                pass
    if not releases:
        raise Exception('Did not find a version of Julia satisfying {!r}'.format(compat.jlstr()))
    return max(releases, key=lambda x: jill.utils.version_utils.Version(x[0]))[0]

def record_resolve(pkgs):
    set_meta("pydeps", {
        "version": __version__,
        "dev": CONFIG["dev"],
        "timestamp": time(),
        "sys_path": sys.path,
        "pkgs": [pkg.dict() for pkg in pkgs],
    })
