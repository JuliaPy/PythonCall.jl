module Conda

import ..External, JSON

const _env = Ref("")

available() = !isempty(_env[])

env() = (env=_env[]; isempty(env) ? error("conda is not available") : env)

create() = External.Conda.runconda(`create -y -p $(env())`)

const ENV_STACK = Vector{Dict{String,String}}()

function activate()
    push!(ENV_STACK, copy(ENV))
    e = env()
    shell = Sys.iswindows() ? "cmd.exe" : "posix"
    pathsep = Sys.iswindows() ? ";" : ":"
    # ask conda for a JSON description of how it would activate the environment
    info = JSON.parse(read(External.Conda._set_conda_env(`$(External.Conda.conda) shell.$shell+json activate $e`), String))
    # make these changes
    # TODO: we currently ignore info["scripts"]["activate"]
    #   run these in a subshell, print the resulting environment vars, and merge into ENV
    #   (or just run the full `conda shell.* activate *` script)
    ENV["PATH"] = join(info["path"]["PATH"], pathsep)
    for k in info["vars"]["unset"]
        delete!(ENV, k)
    end
    ENV["CONDA_PREFIX"] = e
    merge!(ENV, info["vars"]["export"])
    return
end

function deactivate()
    env = pop!(ENV_STACK)
    for k in collect(keys(ENV))
        delete!(ENV, k)
    end
    merge!(ENV, env)
    return
end

python_dir() = External.Conda.python_dir(env())

python_exe() = joinpath(python_dir(), Sys.iswindows() ? "python.exe" : "python")

run(args) = External.Conda.runconda(args, env())

add(pkg; channel="") = External.Conda.add(pkg, env(), channel=channel)

pip_interop(value::Bool) = External.Conda.pip_interop(value, env())

pip_interop() = External.Conda.pip_interop(env())

pip(cmd, pkg=String[]) = External.Conda.pip(cmd, pkg, env())

end
