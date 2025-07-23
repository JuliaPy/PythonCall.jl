import argparse

from juliacall import Main
from juliacall.repl import run_repl, add_repl_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser("JuliaCall REPL (experimental)")
    parser.add_argument('-e', '--eval', type=str, default=None, help='Evaluate <expr>. If specified, all other arguments are ignored.')
    parser.add_argument('-E', '--print', type=str, default=None, help='Evaluate <expr> and display the result. If specified, all other arguments are ignored.')

    add_repl_args(parser)

    args = parser.parse_args()
    assert not (args.eval is not None and args.print is not None), "Cannot specify both -e/--eval and -E/--print"

    if args.eval is not None:
        Main.seval(args.eval)
    elif args.print is not None:
        result = Main.seval(args.print)
        Main.display(result)
    else:
        run_repl(
            banner=args.banner,
            quiet=args.quiet,
            history_file=args.history_file,
            preamble=args.preamble
        )
