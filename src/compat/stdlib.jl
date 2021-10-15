const pymodulehooks = pynew()

function init_stdlib()

    # check word size
    pywordsize = @py(jlbool(pysysmodule.maxsize > 2^32)) ? 64 : 32
    pywordsize == Sys.WORD_SIZE || error("Julia is $(Sys.WORD_SIZE)-bit but Python is $(pywordsize)-bit")

    if !C.CTX.is_embedded

        # set sys.argv
        pysysmodule.argv = pylist([""; ARGS])

        # some modules test for interactivity by checking if sys.ps1 exists
        if isinteractive() && !pyhasattr(pysysmodule, "ps1")
            pysysmodule.ps1 = ">>> "
        end

        # add hook to perform certain actions when certain modules are loaded
        @py g = {}
        @py @exec """
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
        """ g
        pycopy!(pymodulehooks, g["JULIA_COMPAT_HOOKS"])
    end

end
