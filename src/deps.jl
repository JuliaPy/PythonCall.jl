module Deps

import Conda, TOML

### META

const _meta_file = Ref("")

meta_file() = _meta_file[]

function load_meta()
    fn = meta_file()
    isfile(fn) || save_meta(Dict())
    TOML.parsefile(fn)
end

save_meta(meta) = open(meta_file(), "w") do io
    TOML.print(io, meta)
end

function get_meta(keys...)
    meta = load_meta()
    for key in keys
        if haskey(meta, key)
            meta = meta[key]
        else
            return
        end
    end
    meta
end

function set_meta(args...)
    length(args) < 2 && error("setmeta() takes at least 2 arguments")
    here = meta = load_meta()
    for key in args[1:end-2]
        here = get!(Dict{String,Any}, here, key)
    end
    here[args[end-1]] = args[end]
    save_meta(meta)
end

get_dep(package, name) = get_meta("jldeps", package, name)

set_dep(package, name, value) = set_meta("jldeps", package, name, value)

### CONDA

const _conda_env = Ref("")

conda_available() = !isempty(_conda_env[])

function conda_env()
    ans = _conda_env[]
    isempty(ans) && error("Conda is not available")
    return ans
end

conda_script_dir() = Conda.script_dir(conda_env())

conda_create() = Conda.runconda(`create -y -p $(conda_env())`)

const CONDA_ENV_STACK = Vector{Dict{String,String}}()

function conda_activate()
    e = conda_env()
    push!(CONDA_ENV_STACK, copy(ENV))
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
            unsetvarregex = r"^(Remove-Item +\$Env:/|Remove-Variable +)([^ =]+)$"
            runscriptregex = r"^\. +\"(.*)\"$"
        else
            shell = "posix"
            exportvarregex = r"^\\?export ([^ =]+)='(.*)'$"
            setvarregex = r"^([^ =]+)='(.*)'"
            unsetvarregex = r"^\\?unset +([^ ]+)$"
            runscriptregex = r"^\\?\. +\"(.*)\"$"
        end
        for line in eachline(Conda._set_conda_env(`$(Conda.conda) shell.$shell activate $e`))
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
        conda_deactivate()
        rethrow()
    end
end

function conda_deactivate()
    env = pop!(CONDA_ENV_STACK)
    for k in collect(keys(ENV))
        delete!(ENV, k)
    end
    merge!(ENV, env)
    return
end

conda_run(cmd::Cmd) = Conda.runconda(cmd, conda_env())

### PYTHON

python_dir() = Conda.python_dir(conda_env())

python_exe() = joinpath(python_dir(), Sys.iswindows() ? "python.exe" : "python")

### PIP

pip_exe() = joinpath(conda_script_dir(), Sys.iswindows() ? "pip.exe" : "pip")

pip_enable() = Conda.pip_interop(true, conda_env())

function pip_run(cmd::Cmd)
    env = conda_env()
    @info("Running $(`pip $cmd`) in $(env) environment")
    run(Conda._set_conda_env(`$(pip_exe()) $cmd`, env))
    nothing
end

### REQUIRE

function require(func::Function, package, name, value; force=false)
    if force || get_dep(package, name) != value
        func()
        set_dep(package, name, value)
    end
end

function require_conda(package, name, version; channel="", force=false)
    if conda_available()
        value = "$version (conda)"
        if !isempty(channel)
            value = "$value channel=$channel"
        end
        require(package, name, value, force=force) do
            if any(c -> c in ('=', '<', '>'), version)
                spec = "$name $version"
            else
                spec = "$name ==$version"
            end
            cmd = `install -y $spec`
            if !isempty(channel)
                cmd = `$cmd -c $channel`
            end
            conda_run(cmd)
        end
    else
        @warn "Skipping adding conda package (conda not available)" name version channel
    end
end

function require_pip(package, name, version; force=false)
    if conda_available()
        value = "$version (pip)"
        require(package, name, value, force=force) do
            if any(c -> c in ('=', '<', '>'), version)
                spec = "$name $version"
            else
                spec = "$name ==$version"
            end
            pip_run(`install $spec`)
        end
    else
        @warn "Skipping adding pip package (conda not available)" name version
    end
end

end
