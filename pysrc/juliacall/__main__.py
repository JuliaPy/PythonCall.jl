if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser("JuliaCall REPL (experimental)")
    parser.add_argument('-e', '--eval', type=str, default=None, help='Evaluate <expr>. If specified, all other arguments are ignored.')

    parser.add_argument('--banner', choices=['yes', 'no', 'short'], default='yes', help='Enable or disable startup banner')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet startup: no banner, suppress REPL warnings')
    parser.add_argument('--history-file', choices=['yes', 'no'], default='yes', help='Load or save history')
    parser.add_argument('--preamble', type=Path, help='Code to be included before the REPL starts')
    args = parser.parse_args()
    if args.eval is not None:
        from juliacall import Main
        Main.seval(args.eval)
    else:
        from juliacall.repl import run_repl
        run_repl(
            banner=args.banner,
            quiet=args.quiet,
            history_file=args.history_file,
            preamble=args.preamble
        )
