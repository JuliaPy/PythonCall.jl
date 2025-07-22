import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
    print("Experimental JuliaCall REPL requires PYTHON_JULIACALL_HANDLE_SIGNALS=yes")
    exit(1)

from juliacall import Main, Base

Base.is_interactive = True

Main.include(os.path.join(os.path.dirname(__file__), 'banner.jl'))
Main.__PythonCall_banner()

if Main.seval(r'VERSION > v"v1.11.0-alpha1"'):
    no_banner_opt = Base.Symbol("no")
else:
    no_banner_opt = False

Base.run_main_repl(True, False, no_banner_opt, True, True)
