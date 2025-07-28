import argparse
import os
from pathlib import Path

from juliacall import Main, Base

def run_repl(banner='yes', quiet=False, history_file='yes', preamble=None):
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
        print("Experimental JuliaCall REPL requires PYTHON_JULIACALL_HANDLE_SIGNALS=yes")
        exit(1)

    Base.is_interactive = True

    if not quiet:
        Main.include(os.path.join(os.path.dirname(__file__), 'banner.jl'))
        Main.__PythonCall_banner(Base.Symbol(banner))

    if Main.seval(r'VERSION ≥ v"v1.11.0-alpha1"'):
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

def add_repl_args(parser):
    parser.add_argument('--banner', choices=['yes', 'no', 'short'], default='yes', help='Enable or disable startup banner')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet startup: no banner, suppress REPL warnings')
    parser.add_argument('--history-file', choices=['yes', 'no'], default='yes', help='Load or save history')
    parser.add_argument('--preamble', type=Path, help='Code to be included before the REPL starts')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("JuliaCall REPL (experimental)")
    add_repl_args(parser)
    args = parser.parse_args()
    run_repl(args.banner, args.quiet, args.history_file, args.preamble)
