import os
from pathlib import Path
import subprocess
import sys

import pytest


_CHILD = r"""
import os
from pathlib import Path
import shlex
import stat

import juliapkg

juliapkg.resolve()
real_executable = juliapkg.executable()
# Populate the normal JuliaPkg project and libjulia cache before changing only
# the executable result that JuliaCall will see.
juliapkg.project()
juliapkg.libjulia()

launcher = Path(os.environ["JULIACALL_TEST_LAUNCHER"])
mode = os.environ["JULIACALL_TEST_LAUNCHER_MODE"]
if mode == "symlink":
    try:
        launcher.symlink_to(real_executable)
    except OSError as exc:
        print(f"symlink unsupported: {exc}")
        raise SystemExit(77)
elif mode == "wrapper":
    launcher.write_text(
        "#!/bin/sh\nexec " + shlex.quote(real_executable) + ' "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
else:
    raise AssertionError(mode)

juliapkg.executable = lambda: str(launcher)

from juliacall import Main

assert Main.seval("1 + 1") == 2
"""


def _run_launcher_case(
    tmp_path: Path, mode: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "PYTHON_JULIACALL_EXE",
        "PYTHON_JULIACALL_PROJECT",
        "PYTHON_JULIACALL_BINDIR",
        "PYTHON_JULIACALL_LIB",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "pysrc")
    env["JULIACALL_TEST_LAUNCHER"] = str(
        tmp_path / "outside-julia-layout" / "julia"
    )
    env["JULIACALL_TEST_LAUNCHER_MODE"] = mode
    Path(env["JULIACALL_TEST_LAUNCHER"]).parent.mkdir()
    return subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("mode", ["symlink", "wrapper"])
def test_issue_816_launcher_outside_julia_bindir(tmp_path, mode):
    """https://github.com/JuliaPy/PythonCall.jl/issues/816"""
    if mode == "wrapper" and os.name != "posix":
        pytest.skip("the executable wrapper requires POSIX sh")

    result = _run_launcher_case(tmp_path, mode)
    if mode == "symlink" and result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, (
        f"child exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
