module Conda

import Conda as _Conda

const _env = Ref("")

available() = !isempty(_env[])

env() = (env=_env[]; isempty(env) ? error("conda is not available") : env)

create() = _Conda.runconda(`create -y -p $(env())`)

function activate()
    newenv = _Conda._get_conda_env(env())
    for k in collect(keys(ENV))
        delete!(ENV, k)
    end
    merge!(ENV, newenv)
end

python_dir() = _Conda.python_dir(env())

python_exe() = joinpath(python_dir(), Sys.iswindows() ? "python.exe" : "python")

run(args) = _Conda.runconda(args, env())

add(pkg; channel="") = _Conda.add(pkg, env(), channel=channel)

pip_interop(value::Bool) = _Conda.pip_interop(value, env())

pip_interop() = _Conda.pip_interop(env())

pip(cmd, pkg=String[]) = _Conda.pip(cmd, pkg, env())

end
