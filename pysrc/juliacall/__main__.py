if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser("JuliaCall REPL (experimental)")
    parser.add_argument('-e', '--eval', type=str, default=None, help='Evaluate <expr>. If specified, all other arguments are ignored.')
    parser.add_argument('-E', '--print', type=str, default=None, help='Evaluate <expr> and display the result. If specified, all other arguments are ignored.')

    parser.add_argument('--banner', choices=['yes', 'no', 'short'], default='yes', help='Enable or disable startup banner')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet startup: no banner, suppress REPL warnings')
    parser.add_argument('--history-file', choices=['yes', 'no'], default='yes', help='Load or save history')
    parser.add_argument('--preamble', type=Path, help='Code to be included before the REPL starts')
    args = parser.parse_args()
    assert not (args.eval is not None and args.print is not None), "Cannot specify both -e/--eval and -E/--print"
    if args.eval is not None:
        from juliacall import Main
        Main.seval(args.eval)
    elif args.print is not None:
        from juliacall import Main
        result = Main.seval(args.print)
        Main.display(result)
    else:
        from juliacall.repl import run_repl
        run_repl(
            banner=args.banner,
            quiet=args.quiet,
            history_file=args.history_file,
            preamble=args.preamble
        )
