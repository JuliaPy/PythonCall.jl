import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux signal handling")
def test_pytest_shutdown(tmp_path):
    test_file = tmp_path / "test_shutdown.py"
    test_file.write_text('''
import atexit
from juliacall import Main as jl

atexit.register(lambda: print("PYTHON_ATEXIT", flush=True))
jl.seval("""
atexit(() -> println("JULIA_ATEXIT"))
const retained = Ref(0)
finalizer(x -> println("JULIA_FINALIZER"), retained)
""")

def test_gc():
    jl.seval("Threads.@threads for i in 1:100; zeros(10000); GC.gc(); end")
''')
    env = dict(os.environ, PYTHON_JULIACALL_HANDLE_SIGNALS="yes", PYTHON_JULIACALL_THREADS="6")
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "-q", str(test_file)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for marker in ("PYTHON_ATEXIT", "JULIA_ATEXIT", "JULIA_FINALIZER"):
        assert marker in result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="Linux signal handling")
@pytest.mark.parametrize("setting,xoption,plugins,enabled", [
    (None, None, [], True),
    ("no", None, [], True),
    ("yes", None, [], False),
    ("yes", "no", [], True),
    ("no", "yes", [], False),
    ("yes", "", [], True),
    ("yes", "invalid", [], True),
    ("yes", None, ["-p", "no:juliacall"], True),
])
def test_pytest_configuration(tmp_path, setting, xoption, plugins, enabled):
    test_file = tmp_path / "test_configuration.py"
    test_file.write_text(f'''
import faulthandler
import sys

def test_configuration():
    assert "juliacall" not in sys.modules
    assert faulthandler.is_enabled() is {enabled!r}
''')
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTHONFAULTHANDLER", "PYTHON_JULIACALL_HANDLE_SIGNALS"):
        env.pop(name, None)
    if setting is not None:
        env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = setting
    command = [sys.executable]
    if xoption is not None:
        command += ["-X", "juliacall-handle-signals" + ("=" + xoption if xoption else "")]
    command += ["-m", "pytest", "-q", *plugins, str(test_file)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
