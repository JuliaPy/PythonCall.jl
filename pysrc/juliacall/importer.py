import os
import sys

from . import newmodule, Base
from importlib.machinery import ModuleSpec

class Finder:
    def find_spec(self, fullname, path, target=None):
        if path is None:
            path = sys.path
            if '.' in fullname:
                return
            name = fullname
        else:
            name = fullname.split('.')[-1]
        for root in path:
            origin = os.path.join(root, name + '.py.jl')
            if os.path.isfile(origin):
                origin = os.path.realpath(origin)
                return ModuleSpec(fullname, Loader(), origin=origin)

class Loader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        spec = module.__spec__
        name = spec.name
        m = module.__jl_module__ = newmodule(name)
        with open(spec.origin) as fp:
            src = fp.read()
        m.seval("begin;]\n" + src + "\nend")
        ks = [str(k) for k in Base.names(m)]
        ks = [k for k in ks if k != name]
        if not ks:
            ks = [str(k) for k in Base.names(m, all=True)]
            ks = [k for k in ks if not (k == name or k.startswith('_') or '#' in k)]
        for k in ks:
            module.__dict__[k] = getattr(m, k)

sys.meta_path.append(Finder())
