from IPython.core.magic import Magics, magics_class, line_cell_magic
from . import Main, Base, PythonCall

@magics_class
class JuliaMagics(Magics):
    
    @line_cell_magic
    def julia(self, line, cell=None):
        code = line if cell is None else cell
        ans = Main.seval('begin\n' + code + '\nend')
        Base.flush(Base.stdout)
        Base.flush(Base.stderr)
        if not code.strip().endswith(';'):
            return ans

def load_ipython_extension(ip):
    # register magics
    ip.register_magics(JuliaMagics(ip))
    # redirect stdout/stderr
    PythonCall.seval("""begin
        const _redirected_stdout = redirect_stdout()
        const _redirected_stderr = redirect_stderr()
        const _py_stdout = PyIO(pyimport("sys" => "stdout"), buflen=1)
        const _py_stderr = PyIO(pyimport("sys" => "stderr"), buflen=1)
        const _redirect_stdout_task = @async write($_py_stdout, $_redirected_stdout)
        const _redirect_stderr_task = @async write($_py_stderr, $_redirected_stderr)
    end""")
