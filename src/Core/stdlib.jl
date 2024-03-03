const pymodulehooks = pynew()

function init_stdlib()

    # check word size
    pywordsize = pygt(Bool, pysysmodule.maxsize, Int64(2)^32) ? 64 : 32
    pywordsize == Sys.WORD_SIZE || error("Julia is $(Sys.WORD_SIZE)-bit but Python is $(pywordsize)-bit")

    if C.CTX.is_embedded

        # This uses some internals, but Base._start() gets the state more like Julia
        # is if you call the executable directly, in particular it creates workers when
        # the --procs argument is given.
        push!(Base.Core.ARGS, joinpath(ROOT_DIR, "pysrc", "juliacall", "init.jl"))
        Base._start()
        Base.eval(:(PROGRAM_FILE = ""))

        # if Python is interactive, ensure Julia is too
        if pyhasattr(pysysmodule, "ps1")
            Base.eval(:(is_interactive = true))
            REPL = Base.require(Base.PkgId(Base.UUID("3fa0cd96-eef1-5676-8a61-b3b8758bbffb"), "REPL"))
            # adapted from run_main_repl() in julia/base/client.jl
            Base.load_InteractiveUtils()
            term_env = get(ENV, "TERM", Sys.iswindows() ? "" : "dumb")
            term = REPL.Terminals.TTYTerminal(term_env, stdin, stdout, stderr)
            if term.term_type == "dumb"
                repl = REPL.BasicREPL(term)
            else
                repl = REPL.LineEditREPL(term, get(stdout, :color, false), true)
                repl.history_file = false
            end
            pushdisplay(REPL.REPLDisplay(repl))
        end

    else

        # set sys.argv
        pysysmodule.argv = pylist([""; ARGS])

        # some modules test for interactivity by checking if sys.ps1 exists
        if isinteractive() && !pyhasattr(pysysmodule, "ps1")
            pysysmodule.ps1 = ">>> "
        end

        # add hook to perform certain actions when certain modules are loaded
        g = pydict()
        pyexec("""
        import sys
        class JuliaCompatHooks:
            def __init__(self):
                self.hooks = {}
            def find_module(self, name, path=None):
                hs = self.hooks.get(name)
                if hs is not None:
                    for h in hs:
                        h()
            def add_hook(self, name, h):
                if name not in self.hooks:
                    self.hooks[name] = [h]
                else:
                    self.hooks[name].append(h)
                if name in sys.modules:
                    h()
        JULIA_COMPAT_HOOKS = JuliaCompatHooks()
        sys.meta_path.insert(0, JULIA_COMPAT_HOOKS)
        """, g)
        pycopy!(pymodulehooks, g["JULIA_COMPAT_HOOKS"])
    end

end
