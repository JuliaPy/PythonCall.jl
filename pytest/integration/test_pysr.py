def test_integration_pysr():
    "Integration tests for PySR"
    import os
    import platform
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        run_kws = dict(check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "virtualenv", tempdir], **run_kws)

        virtualenv_path = os.path.join(
            tempdir, "Scripts" if platform.system() == "Windows" else "bin"
        )
        virtualenv_executable = os.path.join(virtualenv_path, "python")

        assert os.path.exists(virtualenv_executable)

        # Install this package
        subprocess.run([virtualenv_executable, "-m", "pip", "install", "."], **run_kws)
        # Install PySR with no requirement on JuliaCall
        subprocess.run(
            [virtualenv_executable, "-m", "pip", "install", "--no-deps", "pysr"],
            **run_kws,
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
            **run_kws,
        )
        # Run PySR main test suite
        subprocess.run(
            [virtualenv_executable, "-m", "pysr", "test", "main"], **run_kws,
        )
