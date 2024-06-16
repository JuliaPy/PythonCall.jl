"""Experimental IPython extension for Julia.

Being experimental, it does not form part of the JuliaCall API. It may be changed or removed
in any release.

Enable the extension by calling the magic '%load_ext juliacall'.

Features:
- Magic `%julia code` to evaluate a piece of Julia code in-line.
- Cell magic `%%julia` to evaluate a cell of Julia code.
- Julia's stdin and stdout are redirected to Python's stdin and stdout.
- Calling Julia's 'display(x)' function will display 'x' in IPython.
"""

from IPython.core.magic import Magics, magics_class, line_cell_magic
from . import Main, PythonCall
import __main__

_set_var = Main.seval("(k, v) -> @eval $(Symbol(k)) = $v")
_get_var = Main.seval("k -> hasproperty(Main, Symbol(k)) ? PythonCall.pyjlraw(getproperty(Main, Symbol(k))) : nothing")
_egal = Main.seval("===")

@magics_class
class JuliaMagics(Magics):
    
    @line_cell_magic
    def julia(self, line, cell=None):
        # parse the line and cell into code and variables
        invars = []
        outvars = []
        syncvars = []
        if cell is None:
            code = line
        else:
            code = cell
            for k in line.split():
                if k.startswith('<'):
                    invars.append(k[1:])
                elif k.startswith('>'):
                    outvars.append(k[1:])
                else:
                    syncvars.append(k)
        # copy variables to Julia
        # keep a cache of variables we may want to copy out again
        cachevars = {}
        for k in invars + syncvars:
            if k in __main__.__dict__:
                _set_var(k, __main__.__dict__[k])
                if k in syncvars:
                    cachevars[k] = _get_var(k)
        # run the code
        ans = Main.seval('begin\n' + code + '\nend')
        # flush stderr/stdout
        PythonCall._ipython._flush_stdio()
        # copy variables back to Python
        # only copy those which are new or have changed value
        for k in outvars + syncvars:
            v0 = cachevars.get(k)
            v1 = _get_var(k)
            if v1 is not None and (v0 is None or not _egal(v0, v1)):
                __main__.__dict__[k] = v1._jl_any()
        # return the value unless suppressed with trailing ";"
        if not code.strip().endswith(';'):
            return ans

def load_ipython_extension(ip):
    # register magics
    ip.register_magics(JuliaMagics(ip))
    # redirect stdout/stderr
    if ip.__class__.__name__ == 'TerminalInteractiveShell':
        # no redirection in the terminal
        PythonCall.seval("""module _ipython
            function _flush_stdio()
            end
        end""")
    else:
        # In Julia 1.7+ redirect_stdout() returns a Pipe object. Earlier versions of Julia
        # just return a tuple of the two pipe ends. This is why we have [1] and [2] below.
        # They can be dropped on earlier versions.
        PythonCall.seval("""module _ipython
            using ..PythonCall
            const _redirected_stdout = redirect_stdout()
            const _redirected_stderr = redirect_stderr()
            const _py_stdout = PyIO(pyimport("sys" => "stdout"); line_buffering=true)
            const _py_stderr = PyIO(pyimport("sys" => "stderr"); line_buffering=true)
            const _redirect_stdout_task = @async write($_py_stdout, $_redirected_stdout[1])
            const _redirect_stderr_task = @async write($_py_stderr, $_redirected_stderr[1])
            function _flush_stdio()
                flush(stderr)
                flush(stdout)
                flush(_redirected_stderr[2])
                flush(_redirected_stdout[2])
                flush(_py_stderr)
                flush(_py_stdout)
                nothing
            end
            nothing
        end""")
    ip.events.register('post_execute', PythonCall._ipython._flush_stdio)
    # push displays
    PythonCall.seval("""begin
        pushdisplay(Compat.PythonDisplay())
        pushdisplay(Compat.IPythonDisplay())
        nothing
    end""")
