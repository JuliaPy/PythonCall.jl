def run_repl(banner='yes', quiet=False, history_file='yes', preamble=None):
    import os
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
        print("Experimental JuliaCall REPL requires PYTHON_JULIACALL_HANDLE_SIGNALS=yes")
        exit(1)

    from juliacall import Main, Base

    Base.is_interactive = True

    if not quiet:
        Main.include(os.path.join(os.path.dirname(__file__), 'banner.jl'))
        Main.__PythonCall_banner(Base.Symbol(banner))

    if Main.seval(r'VERSION > v"v1.11.0-alpha1"'):
        no_banner_opt = Base.Symbol("no")
    else:
        no_banner_opt = False

    if preamble:
        Main.include(str(preamble.resolve()))

    Base.run_main_repl(
        Base.is_interactive,
        quiet,
        no_banner_opt,
        history_file == 'yes',
        True
    )

if __name__ == '__main__':
    run_repl()
