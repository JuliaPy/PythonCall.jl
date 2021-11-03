import hashlib
import io
import gzip
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
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

def compatible_julia_versions(compat=None):
    os = get_os()
    arch = get_arch()
    libc = get_libc()
    if libc == '' and os == 'linux':
        warnings.warn('could not determine libc version - assuming glibc')
        libc = 'gnu'
    ans = {}
    for (k, v) in all_julia_versions().items():
        v = v.copy()
        if not v['stable']:
            continue
        files = []
        for f in v['files']:
            assert f['version'] == k
            if not any(f['url'].endswith(ext) for ext in julia_installers):
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

def best_julia_version(compat=None):
    vers = compatible_julia_versions(compat)
    if not vers:
        raise Exception('no compatible Julia version found')
    v = sorted(vers.keys(), key=Version, reverse=True)[0]
    return v, vers[v]

def install_julia(ver, prefix):
    for f in ver['files']:
        url = f['url']
        # find a suitable installer
        installer = None
        for ext in julia_installers:
            if url.endswith(ext):
                installer = julia_installers[ext]
                break
        if installer is None:
            continue
        # download julia
        buf = download_julia(f)
        # include the version in the prefix
        print(f'Installing Julia to {prefix}')
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        if os.path.dirname(prefix):
            os.makedirs(os.path.dirname(prefix))
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
                print(f'  downloaded {buf.tell()/(1<<20):.1f} MB of {size/(1<<20):.1f} MB')
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
    with tempfile.TemporaryDirectory() as tmpdir:
        # extract all files
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(tmpdir)
        # copy stuff out
        srcdirs = [d for d in os.listdir(tmpdir) if d.startswith('julia')]
        if len(srcdirs) != 1:
            raise Exception('expecting one julia* directory')
        shutil.copytree(os.path.join(tmpdir, srcdirs[0]), prefix, symlinks=True)

def install_julia_tar_gz(f, buf, prefix):
    with tempfile.TemporaryDirectory() as tmpdir:
        # extract all files
        with gzip.GzipFile(fileobj=buf) as gf:
            with tarfile.TarFile(fileobj=gf) as tf:
                tf.extractall(tmpdir)
        # copy stuff out
        srcdirs = [d for d in os.listdir(tmpdir) if d.startswith('julia')]
        if len(srcdirs) != 1:
            raise Exception('expecting one julia* directory')
        shutil.copytree(os.path.join(tmpdir, srcdirs[0]), prefix, symlinks=True)

def install_julia_dmg(f, buf, prefix):
    with tempfile.TemporaryDirectory() as tmpdir:
        # write the dmg file out
        dmg = os.path.join(tmpdir, 'dmg')
        with open(dmg, 'wb') as f:
            f.write(buf.read())
        # mount it
        mount = os.path.join(tmpdir, 'mount')
        subprocess.run(['hdiutil', 'mount', '-mount', 'required', '-mountpoint', mount, dmg], check=True, capture_output=True)
        try:
            # copy stuff out
            appdirs = [d for d in os.listdir(mount) if d.startswith('Julia') and d.endswith('.app')]
            if len(appdirs) != 1:
                raise Exception('expecting one Julia*.app directory')
            srcdir = os.path.join(mount, appdirs[0], 'Contents', 'Resources', 'julia')
            shutil.copytree(srcdir, prefix, symlinks=True)
        finally:
            # unmount
            subprocess.run(['umount', mount], check=True, capture_output=True)

julia_installers = {
    '.tar.gz': install_julia_tar_gz,
    '.zip': install_julia_zip,
    '.dmg': install_julia_dmg,
}
