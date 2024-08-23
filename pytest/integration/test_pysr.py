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
                "pytest",
            ],
            check=True,
        )
        # Run PySR main test suite
        subprocess.run(
            [virtualenv_executable, "-m", "pysr", "test", "main"], check=True
        )
