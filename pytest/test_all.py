def test_import():
    import juliacall

def test_issue_394():
    "https://github.com/JuliaPy/PythonCall.jl/issues/394"
    from juliacall import Main as jl
    x = 3
    f = lambda x: x+1
    y = 5
    jl.x = x
    assert jl.x is x
    jl.f = f
    assert jl.f is f
    jl.y = y
    assert jl.y is y
    assert jl.x is x
    assert jl.f is f
    assert jl.y is y
    assert jl.seval("f(x)") == 4


def test_integration_pysr():
    "Integration tests for PySR"
    import os
    import platform
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        subprocess.run([sys.executable, "-m", "virtualenv", tempdir], check=True)

        virtualenv_path = os.path.join(
            tempdir, "Scripts" if platform.system() == "Windows" else "bin"
        )
        virtualenv_executable = os.path.join(virtualenv_path, "python")

        assert os.path.exists(virtualenv_executable)

        # Install this package
        subprocess.run([virtualenv_executable, "-m", "pip", "install", "."], check=True)
        # Install PySR with no requirement on JuliaCall
        subprocess.run(
            [virtualenv_executable, "-m", "pip", "install", "--no-deps", "pysr"],
            check=True,
        )
        # Install PySR test requirements
        subprocess.run(
            [
                virtualenv_executable,
                "-m",
                "pip",
                "install",
                "sympy",
                "pandas",
                "scikit_learn",
                "click",
                "setuptools",
                "typing_extensions",
                "pytest",
                "nbval",
            ],
            check=True,
        )
        # Run PySR main test suite
        subprocess.run(
            [virtualenv_executable, "-m", "pysr", "test", "main"], check=True
        )
