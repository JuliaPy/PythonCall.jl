"""
    python_executable_path()

Path to the Python interpreter, or `missing` if not known.
"""
PythonCall.python_executable_path() = CTX.exe_path

"""
    python_library_path()

Path to libpython, or `missing` if not known.
"""
PythonCall.python_library_path() = CTX.lib_path

"""
    python_library_handle()

Handle to the open libpython, or `C_NULL` if not known.
"""
PythonCall.python_library_handle() = CTX.lib_ptr

"""
    python_version()

The version of Python, or `missing` if not known.
"""
PythonCall.python_version() = CTX.version
