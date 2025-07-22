import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
    print("Experimental JuliaCall REPL requires PYTHON_JULIACALL_HANDLE_SIGNALS=yes")
    exit(1)

from juliacall import Main

Main.seval(f"""\
Base.is_interactive = true

include(\"{os.path.join(os.path.dirname(__file__), 'banner.jl')}\")
banner()

if VERSION > v"v1.11.0-alpha1"
    Base.run_main_repl(true, false, :no, true, true)
else # interactive, quiet, banner, history_file, color_set
    Base.run_main_repl(true, false, false, true, true)
end
""")
