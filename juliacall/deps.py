import os
# import semantic_version as semver
import sys

from time import time

from . import CONFIG, __version__

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

### VERSION PARSING

@semver.base.BaseSpec.register_syntax
class JuliaVersionSpec(semver.SimpleSpec):
    SYNTAX = 'julia'
    class Parser(semver.SimpleSpec.Parser):
        PREFIX_ALIASES = {'=': '==', '': '^'}
        @classmethod
        def parse(cls, expression):
            blocks = expression.split(',')
            clause = semver.base.Never()
            for block in blocks:
                block = block.strip()
                if not cls.NAIVE_SPEC.match(block):
                    raise ValueError('Invalid simple block %r' % block)
                clause |= cls.parse_block(block)
            return clause

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
    import json
    ans = {}
    for fn in deps_files():
        with open(fn) as fp:
            deps = json.load(fp)
        for (name, kw) in deps.get("packages", {}).items():
            if name in ans:
                p = ans[name]
                if p.uuid != kw["uuid"]:
                    raise Exception("found multiple UUIDs for package '{}'".format(name))
                if "dev" in kw:
                    p.dev |= kw["dev"]
                if "compat" in kw:
                    if p.compat is not None and p.compat != kw["compat"]:
                        raise NotImplementedError("found multiple 'compat' entries for package '{}'".format(name))
                    p.compat = kw["compat"]
                if "path" in kw:
                    if p.path is not None and p.path != kw["path"]:
                        raise Exception("found multiple 'path' entries for package '{}'".format(name))
                    p.path = kw["path"]
                if "url" in kw:
                    if p.url is not None and p.url != kw["url"]:
                        raise Exception("found multiple 'url' entries for package '{}'".format(name))
                    p.url = kw["url"]
                if "rev" in kw:
                    if p.rev is not None and p.rev != kw["rev"]:
                        raise Exception("found multiple 'rev' entries for package '{}'".format(name))
                    p.rev = kw["rev"]
                if "version" in kw:
                    if p.version is not None and p.version != kw["version"]:
                        raise NotImplementedError("found multiple 'version' entries for package '{}'".format(name))
                    p.version = kw["version"]
            else:
                p = PackageSpec(name=name, **kw)
                ans[p.name] = p
    return list(ans.values())

def record_resolve(pkgs):
    set_meta("pydeps", {
        "version": __version__,
        "dev": CONFIG["dev"],
        "timestamp": time(),
        "sys_path": sys.path,
        "pkgs": [pkg.dict() for pkg in pkgs],
    })
