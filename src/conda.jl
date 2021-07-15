module Conda

import ..Utils

const _env = Ref("")

available() = !isempty(_env[])

env() = (env=_env[]; isempty(env) ? error("conda is not available") : env)

create() = Utils.Conda.runconda(`create -y -p $(env())`)

function activate()
    e = env()
    # these steps imitate a minimal "conda activate"
    # TODO: run "conda shell.posix activate ..." or "conda shell.cmd.exe activate ...", capture the resulting environment, and set that here
    oldlvl = parse(Int, get(ENV, "CONDA_SHLVL", "0"))
    if oldlvl > 0
        ENV["CONDA_PREFIX_$oldlvl"] = ENV["CONDA_PREFIX"]
    end
    ENV["CONDA_SHLVL"] = string(oldlvl+1)
    ENV["CONDA_PREFIX"] = e
    ENV["CONDA_DEFAULT_ENV"] = e
    ENV["CONDA_PROMPT_MODIFIER"] = "($e) "
    ENV["_CE_M"] = ""
    ENV["_CE_CONDA"] = ""
    ENV["CONDA_PYTHON_EXE"] = python_exe()
    oldpath = get(ENV, "PATH", "")
    pathsep = Sys.iswindows() ? ";" : ":"
    ENV["PATH"] = oldpath == "" ? Utils.Conda.bin_dir(e) : (Utils.Conda.bin_dir(e) * pathsep * oldpath)
    return
end

python_dir() = Utils.Conda.python_dir(env())

python_exe() = joinpath(python_dir(), Sys.iswindows() ? "python.exe" : "python")

run(args) = Utils.Conda.runconda(args, env())

add(pkg; channel="") = Utils.Conda.add(pkg, env(), channel=channel)

pip_interop(value::Bool) = Utils.Conda.pip_interop(value, env())

pip_interop() = Utils.Conda.pip_interop(env())

pip(cmd, pkg=String[]) = Utils.Conda.pip(cmd, pkg, env())

end
