module Conda

import ..External, JSON

const _env = Ref("")

available() = !isempty(_env[])

env() = (env=_env[]; isempty(env) ? error("conda is not available") : env)

create() = External.Conda.runconda(`create -y -p $(env())`)

const ENV_STACK = Vector{Dict{String,String}}()

function activate()
    e = env()
    push!(ENV_STACK, copy(ENV))
    try
        # We run `conda shell.$shell activate $e` to get the shell script which activates
        # the environment, then parse its instructions (taking advantage of the fact that
        # conda outputs these instructions in quite a structured fashion) and make the
        # necessary modifications.
        #
        # Another approach is to use `conda shell.$shell+json activate $e` to get the
        # instructions as JSON, but as of July 2020 the latest version of conda on 32-bit
        # linux does not support +json.
        #
        # Another approach is to actually execute the activate script in a subshell and then
        # extract the resulting environment. This would capture environment variables set
        # in scripts, which are currently ignored, and so replicate the conda environment
        # precisely. Just need to work out how to get at the resulting environment.
        if Sys.iswindows()
            shell = "powershell"
            exportvarregex = r"^\$Env:([^ =]+) *= *\"(.*)\"$"
            setvarregex = exportvarregex
            unsetvarregex = r"^Remove-Item \$Env:/([^ =]+)$"
            runscriptregex = r"^\. +\"(.*)\"$"
        else
            shell = "posix"
            exportvarregex = r"^export ([^ =]+)='(.*)'$"
            setvarregex = r"^([^ =]+)='(.*)'"
            unsetvarregex = r"^unset +([^ ]+)$"
            runscriptregex = r"^\. +\"(.*)\"$"
        end
        for line in eachline(External.Conda._set_conda_env(`$(External.Conda.conda) shell.$shell activate $e`))
            if (m = match(exportvarregex, line)) !== nothing
                @debug "conda activate export var" k=m.captures[1] v=m.captures[2]
                ENV[m.captures[1]] = m.captures[2]
            elseif (m = match(setvarregex, line)) !== nothing
                @debug "ignoring conda activate set var" k=m.captures[1] v=m.captures[2]
            elseif (m = match(unsetvarregex, line)) !== nothing
                @debug "conda activate unset var" k=m.captures[1]
                delete!(ENV, m.captures[1])
            elseif (m = match(runscriptregex, line)) !== nothing
                @debug "ignoring conda activate script" file=m.captures[1]
            elseif !isempty(strip(line))
                @warn "ignoring conda activate line" line
            end
        end
        @assert realpath(ENV["CONDA_PREFIX"]) == realpath(e)
        return
    catch
        deactivate()
        rethrow()
    end
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
