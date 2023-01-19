"""Experimental IPython extension for Julia.

Being experimental, it does not form part of the JuliaCall API. It may be changed or removed
in any release.

Enable the extension by calling the magic '%load_ext juliacall.ipython'.

Features:
- Magic `%jl code` to evaluate a piece of Julia code in-line.
- Cell magic `%%jl` to evaluate a cell of Julia code.
- Julia's stdin and stdout are redirected to Python's stdin and stdout.
- Calling Julia's 'display(x)' function will display 'x' in IPython.
"""

from IPython.core.magic import Magics, magics_class, line_cell_magic
from . import Main, Base, PythonCall

@magics_class
class JuliaMagics(Magics):
    
    @line_cell_magic
    def julia(self, line, cell=None):
        code = line if cell is None else cell
        ans = Main.seval('begin\n' + code + '\nend')
        PythonCall._flush_stdio()
        if not code.strip().endswith(';'):
            return ans

def load_ipython_extension(ip):
    # register magics
    ip.register_magics(JuliaMagics(ip))
    # redirect stdout/stderr
    # push displays
    PythonCall.seval("""begin
        const _redirected_stdout = redirect_stdout()
        const _redirected_stderr = redirect_stderr()
        const _py_stdout = PyIO(pyimport("sys" => "stdout"); line_buffering=true)
        const _py_stderr = PyIO(pyimport("sys" => "stderr"); line_buffering=true)
        const _redirect_stdout_task = @async write($_py_stdout, $_redirected_stdout)
        const _redirect_stderr_task = @async write($_py_stderr, $_redirected_stderr)
        function _flush_stdio()
            flush(stderr)
            flush(stdout)
            flush(_redirected_stderr)
            flush(_redirected_stdout)
            flush(_py_stderr)
            flush(_py_stdout)
            nothing
        end
        pushdisplay(PythonDisplay())
        pushdisplay(IPythonDisplay())
        nothing
    end""")
    ip.events.register('post_execute', PythonCall._flush_stdio)
