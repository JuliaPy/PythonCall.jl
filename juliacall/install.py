import hashlib
import io
import json
import os
import platform
import shutil
import tarfile
import time
import urllib.request
import warnings
import zipfile

from .semver import Version

_all_julia_versions = None

def all_julia_versions():
    global _all_julia_versions
    if _all_julia_versions is None:
        url = 'https://julialang-s3.julialang.org/bin/versions.json'
        print(f'Querying Julia versions from {url}')
        with urllib.request.urlopen(url) as f:
            _all_julia_versions = json.loads(f.read())
    return _all_julia_versions

os_aliases = {
    'darwin': 'mac',
    'windows': 'winnt',
}

def get_os():
    os = platform.system().lower()
    return os_aliases.get(os.lower(), os)

arch_aliases = {
    'arm64': 'aarch64',
    'i386': 'i686',
    'amd64': 'x86_64',
}

def get_arch():
    arch = platform.machine().lower()
    return arch_aliases.get(arch.lower(), arch)

libc_aliases = {
    'glibc': 'gnu',
}

def get_libc():
    libc = platform.libc_ver()[0].lower()
    return libc_aliases.get(libc, libc)

def compatible_julia_versions(compat=None, stable=True, kind=None):
    os = get_os()
    arch = get_arch()
    libc = get_libc()
    if libc == '' and os == 'linux':
        warnings.warn('could not determine libc version - assuming glibc')
        libc = 'gnu'
    ans = {}
    for (k, v) in all_julia_versions().items():
        v = v.copy()
        if stable is not None and v['stable'] != stable:
            continue
        files = []
        for f in v['files']:
            assert f['version'] == k
            if kind is not None and f['kind'] != kind:
                continue
            if f['os'] != os:
                continue
            if f['arch'] != arch:
                continue
            if os == 'linux' and f['triplet'].split('-')[2] != libc:
                continue
            if compat is not None:
                try:
                    ver = Version(f['version'])
                except Exception:
                    continue
                if ver not in compat:
                    continue
            files.append(f)
        if not files:
            continue
        v['files'] = files
        ans[k] = v
    triplets = {f['triplet'] for (k, v) in ans.items() for f in v['files']}
    if len(triplets) > 1:
        raise Exception(f'multiple matching triplets {sorted(triplets)} - this is a bug, please report')
    return ans

def install_julia(vers, prefix):
    if not vers:
        raise Exception('no compatible Julia version found')
    for v in sorted(vers.keys(), key=Version, reverse=True):
        for f in vers[v]['files']:
            url = f['url']
            if url.endswith('.tar.gz'):
                installer = install_julia_tar
            elif url.endswith('.zip'):
                installer = install_julia_zip
            elif url.endswith('.dmg'):
                installer = install_julia_dmg
            else:
                continue
            buf = download_julia(f)
            print(f'Installing Julia to {prefix}')
            if os.path.exists(prefix):
                shutil.rmtree(prefix)
            installer(f, buf, prefix)
            return
    raise Exception('no installable Julia version found')

def download_julia(f):
    url = f['url']
    sha256 = f['sha256']
    size = f['size']
    print(f'Downloading Julia from {url}')
    buf = io.BytesIO()
    freq = 5
    t = time.time() + freq
    with urllib.request.urlopen(url) as f:
        while True:
            data = f.read(1<<16)
            if not data:
                break
            buf.write(data)
            if time.time() > t:
                print(f'  downloaded {buf.tell()/(1<<20):.3f} MB of {size/(1<<20):.3f} MB')
                t = time.time() + freq
    print('  download complete')
    print(f'Verifying download')
    buf.seek(0)
    m = hashlib.sha256()
    m.update(buf.read())
    sha256actual = m.hexdigest()
    if sha256actual != sha256:
        raise Exception(f'SHA-256 hash does not match, got {sha256actual}, expecting {sha256}')
    buf.seek(0)
    return buf

def install_julia_zip(f, buf, prefix):
    os.makedirs(prefix)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(prefix)
    fns = os.listdir(prefix)
    if 'bin' not in fns:
        if len(fns) != 1:
            raise Exception('expecting one subdirectory')
        top = fns[0]
        fns = os.listdir(os.path.join(prefix, top))
        if 'bin' not in fns:
            raise Exception('expecting a bin directory')
        for fn in fns:
            os.rename(os.path.join(prefix, top, fn), os.path.join(prefix, fn))
        os.rmdir(os.path.join(prefix, top))

def install_julia_tar(f, buf, prefix):
    os.makedirs(prefix)
    with tarfile.TarFile(fileobj=buf) as tf:
        tf.extractall(prefix)
    fns = os.listdir(prefix)
    if 'bin' not in fns:
        if len(fns) != 1:
            raise Exception('expecting one subdirectory')
        top = fns[0]
        fns = os.listdir(os.path.join(prefix, top))
        if 'bin' not in fns:
            raise Exception('expecting a bin directory')
        for fn in fns:
            os.rename(os.path.join(prefix, top, fn), os.path.join(prefix, fn))
        os.rmdir(os.path.join(prefix, top))

def install_julia_dmg(f, buf, prefix):
    raise NotImplementedError('cannot yet install Julia from DMG')
