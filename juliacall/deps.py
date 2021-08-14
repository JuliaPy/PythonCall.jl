from . import CONFIG

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
