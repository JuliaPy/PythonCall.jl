import json
import os
import sys
import subprocess

from time import time

from . import CONFIG, __version__
from .semver import JuliaCompat

def julia_version_str(exe):
    """
    If exe is a julia executable, return its version as a string. Otherwise return None.
    """
    try:
        proc = subprocess.run([exe, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        return
    words = proc.stdout.decode('utf-8').split()
    if len(words) < 3 or words[0].lower() != 'julia' or words[1].lower() != 'version':
        return
    return words[2]

### META

META_VERSION = 1 # increment whenever the format changes

def load_meta(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path) as fp:
            meta = json.load(fp)
            if meta.get('meta_version') == META_VERSION:
                return meta

def save_meta(meta_path, meta):
    assert isinstance(meta, dict)
    assert meta.get('meta_version') == META_VERSION
    with open(meta_path, 'w') as fp:
        json.dump(meta, fp)

### RESOLVE

class PackageSpec:
    def __init__(self, name, uuid, dev=False, version=None, path=None, url=None, rev=None):
        self.name = name
        self.uuid = uuid
        self.dev = dev
        self.version = version
        self.path = path
        self.url = url
        self.rev = rev

    def jlstr(self):
        args = ['name="{}"'.format(self.name), 'uuid="{}"'.format(self.uuid)]
        if self.path is not None:
            args.append('path=raw"{}"'.format(self.path))
        if self.url is not None:
            args.append('url=raw"{}"'.format(self.url))
        if self.rev is not None:
            args.append('rev=raw"{}"'.format(self.rev))
        return "Pkg.PackageSpec({})".format(', '.join(args))

    def dict(self):
        ans = {
            "name": self.name,
            "uuid": self.uuid,
            "dev": self.dev,
            "version": self.version,
            "path": self.path,
            "url": self.url,
            "rev": self.rev,
        }
        return {k:v for (k,v) in ans.items() if v is not None}

def can_skip_resolve(isdev_in, meta_path):
    # resolve if we haven't resolved before
    deps = load_meta(meta_path)
    if deps is None:
        return False
    # resolve whenever the version changes
    version = deps.get("version")
    if version is None or version != __version__:
        return False
    # resolve whenever Julia changes
    jlexe = deps.get("jlexe")
    if jlexe is None:
        return False
    jlver = deps.get("jlversion")
    if jlver is None or jlver != julia_version_str(jlexe):
        return False
    # resolve whenever swapping between dev/not dev
    isdev = deps.get("dev")
    if isdev is None or isdev != isdev_in: # CONFIG["dev"]:
        return False
    # resolve whenever anything in sys.path changes
    timestamp = deps.get("timestamp")
    if timestamp is None:
        return False
    timestamp = max(os.path.getmtime(CONFIG["meta"]), timestamp)
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
    return deps

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
            if name == "PythonCall":
                raise ValueError("Cannot have a dependency called 'PythonCall'")
            dep = all_deps.setdefault(name, {})
            for (k, v) in kvs.items():
                if k == 'path':
                    # resolve paths relative to the directory containing the file
                    v = os.path.join(os.path.dirname(fn), v)
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
        merge_compat(kw, kfvs, 'version')
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

def record_resolve(meta_path, pkgs):
    save_meta(meta_path, {
        "meta_version": META_VERSION,
        "version": __version__,
        "dev": CONFIG["dev"],
        "jlversion": CONFIG.get("exever"),
        "jlexe": CONFIG.get("exepath"),
        "timestamp": time(),
        "sys_path": sys.path,
        "pkgs": [pkg.dict() for pkg in pkgs],
    })
