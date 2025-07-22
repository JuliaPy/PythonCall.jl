import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes":
    print("Experimental JuliaCall REPL requires PYTHON_JULIACALL_HANDLE_SIGNALS=yes")
    exit(1)

import argparse
from pathlib import Path
parser = argparse.ArgumentParser("JuliaCall REPL (experimental)")
parser.add_argument('--banner', choices=['yes', 'no', 'short'], default='yes', help='Enable or disable startup banner')
parser.add_argument('--quiet', '-q', action='store_true', help='Quiet startup: no banner, suppress REPL warnings')
parser.add_argument('--history-file', choices=['yes', 'no'], default='yes', help='Load or save history')
parser.add_argument('--preamble', type=Path, help='Code to be included before the REPL starts')
args = parser.parse_args()

from juliacall import Main, Base

Base.is_interactive = True

if not args.quiet:
    Main.include(os.path.join(os.path.dirname(__file__), 'banner.jl'))
    Main.__PythonCall_banner(Base.Symbol(args.banner))

if Main.seval(r'VERSION > v"v1.11.0-alpha1"'):
    no_banner_opt = Base.Symbol("no")
else:
    no_banner_opt = False

if args.preamble:
    Main.include(str(args.preamble.resolve()))

Base.run_main_repl(
    Base.is_interactive,
    args.quiet,
    no_banner_opt,
    args.history_file == 'yes',
    True
)
